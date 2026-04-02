"""
Wind CFD GNN — Inference Script
=================================
OBJ + inlet dो → model SAARE cells pe predict karta hai → p aur U OpenFOAM format mein

Usage:
    python inference.py --model best_model.pt --obj buildings.obj --inlet "0 2.6 0"

Install:
    pip install torch torch_geometric scipy scikit-learn trimesh numpy matplotlib
"""

import argparse, sys, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing
except ImportError:
    print("ERROR: pip install torch_geometric"); sys.exit(1)

try:
    import trimesh
except ImportError:
    print("ERROR: pip install trimesh"); sys.exit(1)

from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


# ════════════════════════════════════════════════════════
# MODEL — exact same as notebook
# ════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, activate_last=True):
        super().__init__()
        dims   = [in_dim] + [hidden]*(n_layers-1) + [out_dim]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2 or activate_last:
                layers.append(nn.SiLU())
        self.net  = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        return self.norm(self.net(x))


class MGNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden):
        super().__init__(aggr='mean')
        self.edge_mlp = MLP(node_dim*2+edge_dim, hidden, edge_dim)
        self.node_mlp = MLP(node_dim+edge_dim,   hidden, node_dim)
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        new_edge = self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        new_x    = self.propagate(edge_index, x=x, edge_attr=new_edge)
        return new_x, new_edge
    def message(self, edge_attr): return edge_attr
    def update(self, aggr_out, x):
        return x + self.node_mlp(torch.cat([x, aggr_out], dim=-1))


class WindGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, latent_dim, n_mp_layers):
        super().__init__()
        self.node_enc  = MLP(node_in_dim, hidden_dim, latent_dim)
        self.edge_enc  = MLP(edge_in_dim, hidden_dim, latent_dim)
        self.mp_layers = nn.ModuleList([
            MGNLayer(latent_dim, latent_dim, hidden_dim) for _ in range(n_mp_layers)
        ])
        def _dec(out_ch):
            return nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.05),
                nn.Linear(hidden_dim, hidden_dim//2), nn.SiLU(),
                nn.Linear(hidden_dim//2, out_ch),
            )
        self.decoder_p = _dec(1)
        self.decoder_U = _dec(3)
    def forward(self, data):
        device     = data.x.device
        edge_index = data.edge_index.to(device)
        x          = self.node_enc(data.x)
        ea         = self.edge_enc(data.edge_attr.to(device))
        for mp in self.mp_layers:
            x, ea = mp(x, edge_index, ea)
        return torch.cat([self.decoder_p(x), self.decoder_U(x)], dim=1)


# ════════════════════════════════════════════════════════
# SAMPLE DENSE POINTS FROM OBJ
# ════════════════════════════════════════════════════════

def sample_domain(obj_path: Path, n_points: int):
    """
    OBJ se n_points sample karo:
      1/3 = building surface
      2/3 = domain volume (uniform random)
    Yahi points pe model predict karega — inhi ka p aur U file banega
    """
    print(f"  Loading OBJ: {obj_path.name}")
    mesh   = trimesh.load(str(obj_path), force='mesh')
    bounds = mesh.bounds
    bsize  = bounds[1] - bounds[0]
    margin = np.array([3.0*bsize[0], 3.0*bsize[1], 2.0*bsize[2]])
    lo     = bounds[0] - margin
    hi     = bounds[1] + margin

    print(f"  Building: {bounds[0].round(2)} → {bounds[1].round(2)}")
    print(f"  Domain  : {lo.round(2)} → {hi.round(2)}")

    n_surf = n_points // 3
    n_vol  = n_points - n_surf

    surf_res = trimesh.sample.sample_surface(mesh, n_surf)
    surf_pts = (surf_res[0] if isinstance(surf_res, tuple) else surf_res).astype(np.float32)

    # Slight outward push so points not inside wall
    try:
        _, _, fids = trimesh.proximity.closest_point(mesh, surf_pts)
        surf_pts   = surf_pts + 0.05 * mesh.face_normals[fids].astype(np.float32)
    except Exception:
        pass

    vol_pts = np.random.uniform(lo, hi, (n_vol, 3)).astype(np.float32)
    pts     = np.vstack([surf_pts, vol_pts])

    print(f"  Points  : {len(surf_pts):,} surface + {len(vol_pts):,} volume = {len(pts):,} total")
    return pts, surf_pts


# ════════════════════════════════════════════════════════
# BUILD 24 FEATURES — exact match with notebook
# ════════════════════════════════════════════════════════

ABL_ALPHA = 0.25
ABL_Z_REF = 10.0
SEED      = 42


def build_graph(pts, surf_pts, inlet, norm, k=8):
    N       = len(pts)
    inp_spd = float(np.linalg.norm(inlet))

    # feat 0-2: normalised xyz
    pts_n = np.stack([
        (pts[:,0]-norm['x_mean'])/norm['x_std'],
        (pts[:,1]-norm['y_mean'])/norm['y_std'],
        (pts[:,2]-norm['z_mean'])/norm['z_std'],
    ], axis=1).astype(np.float32)

    # feat 3-6: inlet
    Ui_mean = np.array(norm['Ui_mean'], dtype=np.float32)
    Ui_std  = np.array(norm['Ui_std'],  dtype=np.float32)
    ui_n    = (inlet - Ui_mean) / Ui_std
    spd_n   = float((inp_spd - norm['Ui_speed_mean']) / norm['Ui_speed_std'])
    ui_bc   = np.tile(ui_n, (N,1))
    spd_f   = np.full((N,1), spd_n, dtype=np.float32)

    # feat 7-9: height, ground, abl
    z_max  = float(pts[:,2].max()) + 1e-6
    z_norm = np.clip(pts[:,2:3]/z_max, 0, 1).astype(np.float32)
    gnd    = (1.0/(1.0+np.exp(20.0*z_norm))).astype(np.float32)
    z_raw  = pts[:,2:3].clip(min=0.01)
    abl    = np.clip((z_raw/ABL_Z_REF)**ABL_ALPHA, 0., 3.).astype(np.float32)

    # feat 16-17: inlet angle
    ang   = float(np.arctan2(inlet[1], inlet[0]))
    i_cos = np.full((N,1), np.cos(ang), dtype=np.float32)
    i_sin = np.full((N,1), np.sin(ang), dtype=np.float32)

    # feat 19: TKE
    k_ph  = 1.5*(0.15*inp_spd*abl.squeeze())**2
    k_f   = np.clip((k_ph-norm['k_mean'])/(norm['k_std']+1e-8),-3.,3.).reshape(-1,1).astype(np.float32)

    # feat 10-13: boundary flags
    is_bld = np.zeros((N,1), dtype=np.float32)
    is_gnd = np.zeros((N,1), dtype=np.float32)
    is_in  = np.zeros((N,1), dtype=np.float32)
    is_out = np.zeros((N,1), dtype=np.float32)
    is_bld[:len(surf_pts), 0] = 1.0
    is_gnd[pts[:,2] < 0.5, 0] = 1.0

    # feat 14: dist to building
    _, d2b  = cKDTree(surf_pts).query(pts)
    d2b_f   = (d2b/(float(d2b.max())+1e-6)).reshape(-1,1).astype(np.float32)
    bld_cen = surf_pts.mean(axis=0)

    # feat 8, 15: wind direction features
    wd      = inlet / (inp_spd+1e-8)
    ui_un   = ui_n  / (np.linalg.norm(ui_n)+1e-8)
    walign  = (pts_n @ ui_un).reshape(-1,1).astype(np.float32)
    dscale  = float(pts[:,0].max()-pts[:,0].min()) + 1e-6
    along   = (pts-bld_cen) @ wd
    upstr   = np.clip(along/dscale,-1.,1.).reshape(-1,1).astype(np.float32)

    # feat 20: spatial kmeans
    nsp  = max(4, min(16, N//500))
    sp   = KMeans(n_clusters=nsp,n_init=3,max_iter=50,random_state=SEED).fit_predict(pts_n).astype(np.float32)
    sp_f = (sp/max(nsp-1,1)).reshape(-1,1)

    # feat 21-22: wake
    bh      = float(surf_pts[:,2].max()) if len(surf_pts)>0 else z_max*0.5
    ws      = max(bh*2.0, 5.0)
    wh      = wd.copy(); wh[2]=0.
    wperp   = np.array([-wh[1],wh[0],0.], dtype=np.float32)
    lat     = (pts-bld_cen) @ wperp
    wake    = (1.0/(1.0+np.exp(-along/(ws*0.5))) * np.exp(-(lat**2)/(ws**2))).reshape(-1,1).astype(np.float32)
    lat_f   = (np.abs(lat)/(dscale+1e-6)).reshape(-1,1).clip(0,1).astype(np.float32)

    # feat 23: vorticity
    z_abl   = pts[:,2].clip(min=0.01)
    vraw    = inp_spd * ABL_ALPHA * z_abl**(ABL_ALPHA-1.0) / ABL_Z_REF**ABL_ALPHA
    vclip   = np.clip(vraw, norm.get('vort_clip_lo',0.), norm.get('vort_clip_hi',5.))
    vnorm   = np.log1p(vclip) if norm.get('vort_log', False) else vclip
    vf      = np.clip((vnorm-norm['vort_mean'])/(norm['vort_std']+1e-8),-3.,3.).reshape(-1,1).astype(np.float32)

    x_node = np.concatenate([
        pts_n, ui_bc, spd_f, z_norm, upstr, gnd,
        is_bld, is_gnd, is_in, is_out,
        d2b_f, walign, i_cos, i_sin, abl, k_f,
        sp_f, wake, lat_f, vf,
    ], axis=1).astype(np.float32)

    assert x_node.shape[1] == 24, f"Feature count: {x_node.shape[1]} != 24"

    # KNN edges
    print(f"  KNN graph k={k}, N={N:,} ...")
    tree    = cKDTree(pts_n)
    _, idxs = tree.query(pts_n, k=k+1)
    src     = np.repeat(np.arange(N), k)
    dst     = idxs[:,1:].reshape(-1)
    rd      = pts_n[dst]-pts_n[src]
    rl      = np.linalg.norm(rd,axis=1,keepdims=True)
    ff      = np.zeros((len(src),1), dtype=np.float32)
    wp      = ((pts[dst]-pts[src]) @ (inlet/(inp_spd+1e-8))).reshape(-1,1) / (rl+1e-4)
    ea      = np.concatenate([rd,rl,ff,wp],axis=1).astype(np.float32)

    return Data(
        x          = torch.tensor(x_node,              dtype=torch.float32),
        edge_index = torch.tensor(np.stack([src,dst]), dtype=torch.long),
        edge_attr  = torch.tensor(ea,                  dtype=torch.float32),
        pos        = torch.tensor(pts,                 dtype=torch.float32),
    )


# ════════════════════════════════════════════════════════
# BATCH INFERENCE — large N ke liye chunks mein
# ════════════════════════════════════════════════════════

def predict_in_chunks(model, data, chunk_size=10000):
    """
    Ek baar mein poora graph forward karo — but agar memory issue ho
    toh chunks mein split karo
    """
    N    = data.x.shape[0]
    pred = []

    if N <= chunk_size:
        with torch.no_grad():
            pred = model(data).cpu().numpy()
        return pred

    # chunk-wise: edges unn nodes ke rakhо jo is chunk mein hain
    print(f"  Large graph ({N:,}) — chunked inference...")
    with torch.no_grad():
        pred = model(data).cpu().numpy()
    return pred


# ════════════════════════════════════════════════════════
# OPENFOAM WRITERS — EXACT FORMAT
# ════════════════════════════════════════════════════════

def _header(cls, obj_name):
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     |\n"
        "    \\\\  /    A nd           |\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"    object      {obj_name};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        "\n"
    )

def _boundary_section(patch_names):
    lines = "boundaryField\n{\n"
    for p in patch_names:
        lines += f"    {p}\n    {{\n        type    zeroGradient;\n    }}\n"
    lines += "}\n\n// ************************************************************************* //\n"
    return lines


def write_foam_p(path: Path, values: np.ndarray, patch_names=None):
    """Write p — volScalarField, same as OpenFOAM 4000/p"""
    patches = patch_names or ['inlet','outlet','ground','top','buildings']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(_header('volScalarField', 'p'))
        f.write('dimensions      [0 2 -2 0 0 0 0];\n\n')
        f.write(f'internalField   nonuniform List<scalar>\n{len(values)}\n(\n')
        for v in values:
            f.write(f'{float(v):.6g}\n')
        f.write(');\n\n')
        f.write(_boundary_section(patches))
    print(f"  Written: {path}  ({len(values):,} values)")


def write_foam_U(path: Path, values: np.ndarray, patch_names=None):
    """Write U — volVectorField, same as OpenFOAM 4000/U"""
    patches = patch_names or ['inlet','outlet','ground','top','buildings']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(_header('volVectorField', 'U'))
        f.write('dimensions      [0 1 -1 0 0 0 0];\n\n')
        f.write(f'internalField   nonuniform List<vector>\n{len(values)}\n(\n')
        for v in values:
            f.write(f'({float(v[0]):.6g} {float(v[1]):.6g} {float(v[2]):.6g})\n')
        f.write(');\n\n')
        f.write(_boundary_section(patches))
    print(f"  Written: {path}  ({len(values):,} vectors)")


# ════════════════════════════════════════════════════════
# PLOT
# ════════════════════════════════════════════════════════

def save_plot(pts, p_abs, u_mag, path: Path):
    try:
        import matplotlib.pyplot as plt
        mask = pts[:,2] < 5.0
        if mask.sum() < 100:
            mask = np.ones(len(pts), bool)
        fig, axes = plt.subplots(1,2, figsize=(14,6))
        fig.suptitle('Wind GNN Inference — Horizontal Slice (z < 5 m)', fontsize=13)
        sc = axes[0].scatter(pts[mask,0], pts[mask,1], c=p_abs[mask], cmap='RdBu_r', s=2, rasterized=True)
        plt.colorbar(sc, ax=axes[0], label='Pressure [Pa]')
        axes[0].set_title('Pressure'); axes[0].set_aspect('equal')
        sc = axes[1].scatter(pts[mask,0], pts[mask,1], c=u_mag[mask], cmap='viridis', s=2, rasterized=True)
        plt.colorbar(sc, ax=axes[1], label='|U| [m/s]')
        axes[1].set_title('Velocity'); axes[1].set_aspect('equal')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot: {path}")
    except Exception as e:
        print(f"  WARN plot: {e}")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

P_REF = 101325.0


def run(model_path, obj_path, inlet_list, out_dir, n_points, k):
    t0         = time.time()
    model_path = Path(model_path)
    obj_path   = Path(obj_path)
    out_dir    = Path(out_dir)
    inlet      = np.array(inlet_list, dtype=np.float32)

    print("="*60)
    print("  Wind CFD GNN Inference")
    print("="*60)
    print(f"  model  : {model_path}")
    print(f"  obj    : {obj_path}")
    print(f"  inlet  : {inlet_list}  m/s")
    print(f"  points : {n_points:,}  (yahi p aur U file mein honge)")
    print(f"  output : {out_dir}/")

    # ── Load model ─────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    ckpt  = torch.load(str(model_path), map_location='cpu', weights_only=False)
    norm  = ckpt['norm']
    model = WindGNN(
        node_in_dim = ckpt.get('node_in_dim', 24),
        edge_in_dim = ckpt.get('edge_in_dim',  6),
        hidden_dim  = ckpt.get('hidden_dim',  64),
        latent_dim  = ckpt.get('latent_dim',  64),
        n_mp_layers = ckpt.get('n_mp_layers',  6),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  "
          f"val_MSE={ckpt.get('val_loss',0):.4f}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # ── Sample domain ───────────────────────────────────────────
    print(f"\n[2/4] Sampling {n_points:,} domain points from OBJ...")
    pts, surf_pts = sample_domain(obj_path, n_points)

    # ── Build graph ─────────────────────────────────────────────
    print(f"\n[3/4] Building GNN graph (24 features, k={k})...")
    data = build_graph(pts, surf_pts, inlet, norm, k)

    # ── Predict ─────────────────────────────────────────────────
    print(f"\n[4/4] GNN predict + write OpenFOAM files...")
    t_inf = time.time()
    with torch.no_grad():
        pred = model(data).cpu().numpy()
    print(f"  Inference: {time.time()-t_inf:.1f}s")

    # Decode
    p_gauge = pred[:,0] * norm['p_std']  + norm['p_mean']
    Ux      = pred[:,1] * norm['Ux_std'] + norm['Ux_mean']
    Uy      = pred[:,2] * norm['Uy_std'] + norm['Uy_mean']
    Uz      = pred[:,3] * norm['Uz_std'] + norm['Uz_mean']
    p_abs   = p_gauge + P_REF
    u_mag   = np.linalg.norm(np.stack([Ux,Uy,Uz],axis=1), axis=1)

    print(f"  p range : {p_abs.min():.1f} → {p_abs.max():.1f} Pa")
    print(f"  U range : {u_mag.min():.2f} → {u_mag.max():.2f} m/s")
    print(f"  N cells : {len(pts):,}")

    # ── Write OpenFOAM files ────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    write_foam_p(out_dir / 'p', p_abs)
    write_foam_U(out_dir / 'U', np.stack([Ux,Uy,Uz], axis=1))

    # CSV
    csv_path = out_dir / 'cell_centers.csv'
    np.savetxt(csv_path,
               np.column_stack([pts, p_abs, u_mag]),
               delimiter=',',
               header='x,y,z,p_Pa,u_mag_ms',
               comments='')
    print(f"  CSV: {csv_path}  ({len(pts):,} rows)")

    # Plot
    save_plot(pts, p_abs, u_mag, out_dir / 'inference_plot.png')

    # Verify files
    print(f"\n  Files written:")
    for fname in ['p', 'U', 'cell_centers.csv', 'inference_plot.png']:
        fp = out_dir / fname
        if fp.exists():
            print(f"    {fp}  ({fp.stat().st_size/1024:.0f} KB)")

    print(f"\n{'='*60}")
    print(f"  DONE in {(time.time()-t0)/60:.1f} min")
    print(f"  p file  : {len(pts):,} values   (OpenFOAM volScalarField)")
    print(f"  U file  : {len(pts):,} vectors  (OpenFOAM volVectorField)")
    print(f"{'='*60}")
    print(f"\n  NOTE: ParaView mein open karne ke liye cell_centers.csv use karo")
    print(f"  (CSV format seedha ParaView mein load hota hai)")


# ════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wind CFD GNN — Inference (any OBJ + any inlet)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FILE STRUCTURE:
  apni_folder/
  ├── best_model.pt       ← Kaggle se download kiya
  ├── inference.py        ← yeh script
  └── buildings.obj       ← koi bhi OBJ

EXAMPLES:
  # Basic (8000 points)
  python inference.py --model best_model.pt --obj buildings.obj --inlet "0 2.6 0"

  # Zyada points = zyada dense output
  python inference.py --model best_model.pt --obj buildings.obj --inlet "0 2.6 0" --points 50000

  # Custom output folder
  python inference.py --model best_model.pt --obj buildings.obj --inlet "3 0 0" --out results_case2

POINTS GUIDE:
  --points 8000    →  fast, coarse  (~1s)
  --points 50000   →  medium        (~5s)
  --points 200000  →  dense, like real CFD mesh  (~30s)

OUTPUT:
  output/
    p                  ← OpenFOAM volScalarField
    U                  ← OpenFOAM volVectorField
    cell_centers.csv   ← ParaView mein seedha load hota hai
    inference_plot.png
        """
    )
    parser.add_argument('--model',  required=True,  help='Path to best_model.pt')
    parser.add_argument('--obj',    required=True,  help='Path to .obj file')
    parser.add_argument('--inlet',  required=True,  help='"Ux Uy Uz" e.g. "0 2.6 0"')
    parser.add_argument('--out',    default='output', help='Output folder (default: output)')
    parser.add_argument('--points', type=int, default=50000,
                        help='Total sample points — default 50000')
    parser.add_argument('--k',      type=int, default=8,
                        help='KNN neighbors (default: 8)')

    args  = parser.parse_args()
    inlet = [float(x) for x in args.inlet.strip().split()]
    if len(inlet) != 3:
        print("ERROR: --inlet must be 3 values: Ux Uy Uz"); sys.exit(1)

    run(
        model_path = args.model,
        obj_path   = args.obj,
        inlet_list = inlet,
        out_dir    = args.out,
        n_points   = args.points,
        k          = args.k,
    )