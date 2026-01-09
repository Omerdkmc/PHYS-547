#!/usr/bin/env python3
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# =========================
#  Ising utilities
# =========================

def initialize_lattice(N):
    return np.random.choice([-1, 1], size=(N, N))

def energy(lattice):
    N = lattice.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            E -= lattice[i, j] * (lattice[(i+1) % N, j] + lattice[i, (j+1) % N])
    return E / lattice.size

def magnetization(lattice):
    return np.sum(lattice) / lattice.size

def metropolis_algorithm(lattice, T):
    N = lattice.shape[0]
    sx, sy = np.random.randint(0, N, size=2)
    dE = 2 * lattice[sx, sy] * (
        lattice[(sx+1) % N, sy] + lattice[(sx-1) % N, sy] +
        lattice[sx, (sy+1) % N] + lattice[sx, (sy-1) % N]
    )
    if dE <= 0 or np.random.rand() < np.exp(-dE / T):
        lattice[sx, sy] *= -1
    return lattice

# =========================
# Cluster recognition via flood fill
# =========================

def cluster_areas_flood_fill(spins):
    """
    Floodfill (DFS) connected-component labeling on a 2D ±1 lattice.
    4-neighbor connectivity with periodic boundaries.

    Returns:
      areas: 1D np.array of cluster sizes (number of sites in each cluster)
    """
    L = spins.shape[0]
    visited = np.zeros((L, L), dtype=bool)
    areas = []

    for i0 in range(L):
        for j0 in range(L):
            if visited[i0, j0]:
                continue

            spin0 = spins[i0, j0]
            stack = [(i0, j0)]
            visited[i0, j0] = True
            area = 0

            while stack:
                i, j = stack.pop()
                area += 1

                # periodic neighbors
                inb = (i - 1) % L
                ipb = (i + 1) % L
                jnb = (j - 1) % L
                jpb = (j + 1) % L

                neigh = ((inb, j), (ipb, j), (i, jnb), (i, jpb))
                for ni, nj in neigh:
                    if (not visited[ni, nj]) and (spins[ni, nj] == spin0):
                        visited[ni, nj] = True
                        stack.append((ni, nj))

            areas.append(area)

    return np.array(areas, dtype=np.int32)

def domain_length_from_clusters_floodfill(spins):
    """
    Convert cluster areas -> a single typical domain length L_cl.

    We use:
      R_c = sqrt(A_c/pi)  (equivalent disk radius)
      L_cl = sum(A_c * R_c) / sum(A_c)  (area-weighted mean)
    """
    areas = cluster_areas_flood_fill(spins).astype(np.float64)
    R = np.sqrt(areas / np.pi)
    L_cl = np.sum(areas * R) / np.sum(areas)
    return float(L_cl)

# =========================
#  (ii) Structure factor length (first moment)
# =========================

def length_from_structure_factor_first_moment(
    spins, *, use_2pi=True, subtract_mean=False, qmax_frac=0.3
):
    """
    L = 2π/<k> (or 1/<k>), where <k> = sum k S(k) / sum S(k),
    using shell-averaged S(k) up to a cutoff qmax_frac*(Nyquist).
    """
    N = spins.shape[0]
    x = spins.astype(np.float64)
    if subtract_mean:
        x = x - x.mean()

    F = np.fft.fft2(x)
    S = (np.abs(F)**2) / (N*N)

    # Remove DC mode
    S[0, 0] = 0.0

    # integer Fourier indices (robustly rounded)
    qx = np.rint(np.fft.fftfreq(N) * N).astype(np.int32)
    qy = np.rint(np.fft.fftfreq(N) * N).astype(np.int32)
    QX, QY = np.meshgrid(qx, qy, indexing="ij")
    q2 = (QX*QX + QY*QY).astype(np.int64)

    q_nyq = N // 2
    qmax = max(1, int(qmax_frac * q_nyq))

    # exclude q2==0 and apply cutoff
    mask = (q2 > 0) & (q2 <= qmax*qmax)

    q2_vals = q2[mask].ravel()
    S_vals  = S[mask].ravel()

    uq2, inv = np.unique(q2_vals, return_inverse=True)
    S_shell = np.zeros_like(uq2, dtype=np.float64)
    np.add.at(S_shell, inv, S_vals)

    k_shell = (2.0*np.pi/N) * np.sqrt(uq2.astype(np.float64))

    denom = S_shell.sum()
    if denom <= 0:
        return 0.0

    k_mean = (k_shell * S_shell).sum() / denom
    if k_mean <= 0:
        return 0.0

    return float((2*np.pi / k_mean) if use_2pi else (1.0 / k_mean))

# =========================
#  Equilibrium simulation
# =========================

def simulate_ising(N, T, *, tol=0.01, window=1000, max_sweeps=50000,
                   meas_sweeps=5000):
    # --- size-scaled sweep counts ---
    N_ref = 64
    z = 1.0
    factor = (N / N_ref)**z

    t_eq_min = int(0.5 * max_sweeps * factor)
    t_eq_max = int(1.0 * max_sweeps * factor)
    t_meas   = int(meas_sweeps * factor)

    lattice = initialize_lattice(N)

    M_record = []
    sweep = 0
    equilibrium_reached = False

    while sweep < t_eq_max and not equilibrium_reached:
        for _ in range(N*N):
            lattice = metropolis_algorithm(lattice, T)
        sweep += 1

        M_record.append(abs(magnetization(lattice)))

        if sweep < t_eq_min or sweep < 2*window:
            continue

        recent = np.array(M_record[-window:], float)
        prev   = np.array(M_record[-2*window:-window], float)

        mean_recent = recent.mean()
        mean_prev   = prev.mean()
        delta = abs(mean_recent - mean_prev) / (abs(mean_prev) + 1e-12)

        if delta < tol:
            equilibrium_reached = True

    M_values, E_values = [], []
    for _ in range(t_meas):
        for __ in range(N*N):
            lattice = metropolis_algorithm(lattice, T)
        m = magnetization(lattice)
        M_values.append(abs(m))
        E_values.append(energy(lattice))

    M_values = np.asarray(M_values, float)
    E_values = np.asarray(E_values, float)

    return float(M_values.mean()), float(E_values.mean())

# =========================
#  Domain growth simulation
# =========================

def make_log_sample_sweeps(tmax, nsamples, tmin=1):
    """
    Log-spaced integer sweep times in [tmin, tmax].
    Ensures uniqueness and sorting.
    """
    tmin = max(1, int(tmin))
    tmax = max(tmin, int(tmax))
    if nsamples <= 1:
        return np.array([tmax], dtype=np.int32)

    xs = np.logspace(np.log10(tmin), np.log10(tmax), nsamples)
    sweeps = np.unique(xs.astype(np.int32))
    sweeps = sweeps[(sweeps >= tmin) & (sweeps <= tmax)]
    if sweeps.size == 0 or sweeps[0] != tmin:
        sweeps = np.unique(np.concatenate(([tmin], sweeps)))
    if sweeps[-1] != tmax:
        sweeps = np.unique(np.concatenate((sweeps, [tmax])))
    return sweeps.astype(np.int32)

def simulate_domain_growth(N, T, *, tmax=5000, sample_sweeps=None,
                          growth_metric="floodfill",
                          sf_use_2pi=False, sf_subtract_mean=False,
                          sf_qmax_frac=0.3):
    """
    Quench protocol:
      start random (T=∞)
      evolve at target T
      measure a length scale L(t) and |m(t)| at sample_sweeps.

    Returns:
      t_series (float), L_series (float), Mabs_series (float)
    """
    lattice = initialize_lattice(N)

    if sample_sweeps is None:
        sample_sweeps = make_log_sample_sweeps(tmax=tmax, nsamples=30, tmin=1)

    sample_sweeps = np.asarray(sample_sweeps, dtype=np.int32)
    L_series = np.empty(len(sample_sweeps), dtype=np.float64)
    Mabs_series = np.empty(len(sample_sweeps), dtype=np.float64)

    k = 0
    for sweep in range(1, int(tmax) + 1):
        for _ in range(N*N):
            lattice = metropolis_algorithm(lattice, T)

        if k < len(sample_sweeps) and sweep == sample_sweeps[k]:
            Mabs_series[k] = abs(magnetization(lattice))

            if growth_metric == "floodfill":
                L_series[k] = domain_length_from_clusters_floodfill(lattice)
            elif growth_metric == "sf":
                L_series[k] = length_from_structure_factor_first_moment(
                    lattice,
                    use_2pi=sf_use_2pi,
                    subtract_mean=sf_subtract_mean,
                    qmax_frac=sf_qmax_frac
                )
            else:
                raise ValueError(f"Unknown growth_metric: {growth_metric}")

            k += 1
            if k == len(sample_sweeps):
                break

    return sample_sweeps.astype(np.float64), L_series, Mabs_series

# =========================
#  MPI scaffolding
# =========================

_TAG_TASK, _TAG_RESULT, _TAG_STOP = 1, 2, 9

def parse_args():
    ap = argparse.ArgumentParser(description="MPI-parallel 2D Ising: equilibrium or domain growth")
    ap.add_argument("--mode", choices=["equilibrium", "growth"], default="equilibrium",
                    help="equilibrium: compute <|M|>, <E>; growth: compute L(t) and |m(t)|")

    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--Tmin", type=float, default=1.5)
    ap.add_argument("--Tmax", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--reps", type=int, default=10, help="Replicates per temperature")

    # equilibrium params
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--max_sweeps", type=int, default=50000)
    ap.add_argument("--meas_sweeps", type=int, default=5000)

    # growth params
    ap.add_argument("--tmax", type=int, default=5000, help="Max sweeps for domain growth (growth mode)")
    ap.add_argument("--nsamples", type=int, default=35, help="How many log-spaced time samples (growth mode)")
    ap.add_argument("--tmin_sample", type=int, default=1, help="Min sweep to sample (growth mode)")

    # choose growth metric
    ap.add_argument("--growth_metric", choices=["floodfill", "sf"], default="floodfill",
                    help="growth mode length estimator: floodfill (clusters) or sf (structure factor)")
    ap.add_argument("--sf_use_2pi", action="store_true",
                    help="if growth_metric=sf, define L=2π/<k> instead of 1/<k>")
    ap.add_argument("--sf_subtract_mean", action="store_true",
                    help="if growth_metric=sf, subtract mean spin before FFT (optional)")
    ap.add_argument("--sf_qmax_frac", type=float, default=0.3,
                    help="if growth_metric=sf, cutoff fraction of Nyquist (e.g. 0.3-0.5)")

    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", type=str, default="data")
    return ap.parse_args()

def worker(rank, base_seed):
    comm = MPI.COMM_WORLD
    while True:
        status = MPI.Status()
        msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == _TAG_STOP:
            break
        if status.Get_tag() != _TAG_TASK:
            continue

        (mode, i, T, rep, N,
         tol, window, max_sweeps, meas_sweeps,
         tmax, sample_sweeps,
         growth_metric, sf_use_2pi, sf_subtract_mean, sf_qmax_frac) = msg

        # deterministic per-(i,rep)
        seed = (np.uint32(base_seed)
                + np.uint32(1000003*rep)
                + np.uint32(9721*i)) % np.uint32(2**32 - 1)
        np.random.seed(int(seed))

        if mode == "equilibrium":
            M, E = simulate_ising(N=N, T=T, tol=tol, window=window,
                                  max_sweeps=max_sweeps, meas_sweeps=meas_sweeps)
            comm.send(("equilibrium", i, rep, float(T), float(M), float(E)),
                      dest=0, tag=_TAG_RESULT)
        else:
            t_series, L_series, Mabs_series = simulate_domain_growth(
                N=N, T=T, tmax=tmax, sample_sweeps=sample_sweeps,
                growth_metric=growth_metric,
                sf_use_2pi=sf_use_2pi,
                sf_subtract_mean=sf_subtract_mean,
                sf_qmax_frac=sf_qmax_frac
            )
            comm.send(("growth", i, rep, float(T), L_series.tolist(), Mabs_series.tolist()),
                      dest=0, tag=_TAG_RESULT)

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    os.makedirs(args.outdir, exist_ok=True)

    temps = np.array([args.Tmin] if args.steps <= 1 else
                     np.linspace(args.Tmin, args.Tmax, args.steps), dtype=float)

    # for growth mode, define global sample times once (same for all reps/temps)
    if args.mode == "growth":
        sample_sweeps = make_log_sample_sweeps(tmax=args.tmax, nsamples=args.nsamples, tmin=args.tmin_sample)
    else:
        sample_sweeps = None

    if rank == 0:
        tasks = [(i, float(temps[i]), rep) for i in range(len(temps)) for rep in range(args.reps)]
        next_task = 0
        inflight = 0

        if args.mode == "equilibrium":
            M_lists = [[] for _ in range(len(temps))]
            E_lists = [[] for _ in range(len(temps))]
        else:
            L_lists = [[] for _ in range(len(temps))]
            Mabs_lists = [[] for _ in range(len(temps))]

        # prime workers
        for w in range(1, size):
            if next_task >= len(tasks):
                break
            i, T, rep = tasks[next_task]
            next_task += 1
            inflight += 1
            comm.send((args.mode, i, T, rep, args.N,
                       args.tol, args.window, args.max_sweeps, args.meas_sweeps,
                       args.tmax, sample_sweeps,
                       args.growth_metric, args.sf_use_2pi, args.sf_subtract_mean, args.sf_qmax_frac),
                      dest=w, tag=_TAG_TASK)

        status = MPI.Status()
        while inflight > 0:
            payload = comm.recv(source=MPI.ANY_SOURCE, tag=_TAG_RESULT, status=status)
            src = status.Get_source()
            inflight -= 1

            kind = payload[0]
            if kind == "equilibrium":
                _, i, rep, T, M, E = payload
                M_lists[i].append(M)
                E_lists[i].append(E)
                completed_for_T = len(M_lists[i])
                done_tasks = sum(len(lst) for lst in M_lists)
            else:
                _, i, rep, T, L_series, Mabs_series = payload
                L_lists[i].append(np.array(L_series, dtype=float))
                Mabs_lists[i].append(np.array(Mabs_series, dtype=float))
                completed_for_T = len(L_lists[i])
                done_tasks = sum(len(lst) for lst in L_lists)

            total_tasks = len(tasks)
            print(f"[{done_tasks}/{total_tasks}] mode={args.mode} T={T:.3f} rep={rep} -> {completed_for_T}/{args.reps} completed for T", flush=True)
            if completed_for_T == args.reps:
                print(f"--> Temperature {T:.3f} finished all {args.reps} replicates", flush=True)

            # feed next task
            if next_task < len(tasks):
                i2, T2, rep2 = tasks[next_task]
                next_task += 1
                inflight += 1
                comm.send((args.mode, i2, T2, rep2, args.N,
                           args.tol, args.window, args.max_sweeps, args.meas_sweeps,
                           args.tmax, sample_sweeps,
                           args.growth_metric, args.sf_use_2pi, args.sf_subtract_mean, args.sf_qmax_frac),
                          dest=src, tag=_TAG_TASK)

        # stop workers
        for w in range(1, size):
            comm.send(None, dest=w, tag=_TAG_STOP)

        # =========================
        #  OUTPUT / AGGREGATION
        # =========================
        if args.mode == "equilibrium":
            M_avg = np.empty(len(temps))
            M_err = np.empty(len(temps))
            E_avg = np.empty(len(temps))
            E_err = np.empty(len(temps))

            for i in range(len(temps)):
                M_arr = np.array(M_lists[i], float)
                E_arr = np.array(E_lists[i], float)

                M_avg[i] = np.mean(M_arr)
                E_avg[i] = np.mean(E_arr)

                M_err[i] = M_arr.std(ddof=1) / np.sqrt(len(M_arr))
                E_err[i] = E_arr.std(ddof=1) / np.sqrt(len(E_arr))

            np.savetxt(
                os.path.join(args.outdir, f"ising_data_parallel_N{args.N}_reps{args.reps}_T{args.Tmin}-{args.Tmax}.txt"),
                np.column_stack((temps, M_avg, M_err, E_avg, E_err)),
                header="Temperature\tMean_|M|\tSE_|M|\tMean_E_per_Spin\tSE_E"
            )

            plt.figure(figsize=(6, 4.5))
            plt.errorbar(temps, M_avg, yerr=M_err, fmt='o', capsize=3, elinewidth=1, markersize=4)
            plt.axvline(2.269, ls='--', label=r'$T_c \approx 2.269$')
            plt.xlabel("Temperature T")
            plt.ylabel(r"$\langle |M| \rangle$ (mean ± SE)")
            plt.legend(loc="best", frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"magnetization_vs_T_N{args.N}_reps{args.reps}_T{args.Tmin}-{args.Tmax}.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(6, 4.5))
            plt.errorbar(temps, E_avg, yerr=E_err, fmt='s', capsize=3, elinewidth=1, markersize=4)
            plt.xlabel("Temperature T")
            plt.ylabel(r"$\langle E \rangle$ per spin (mean ± SE)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"energy_vs_T_N{args.N}_reps{args.reps}_T{args.Tmin}-{args.Tmax}.png"), dpi=200)
            plt.close()

            for T, M, E in zip(temps, M_avg, E_avg):
                print(f"T={T:.3f} → mean⟨|M|⟩={M:+.4f}, mean⟨E⟩/spin={E:+.4f}")

        else:
            # growth mode aggregation
            t = sample_sweeps.astype(float)

            metric = args.growth_metric
            yname = "Lcl" if metric == "floodfill" else "Lsf"
            ylab  = (r"Domain size $L_{\mathrm{cl}}(t)$ (flood fill)"
                     if metric == "floodfill"
                     else (r"Length $L_S(t)$ from structure factor"
                           + (r" ($2\pi/\langle k\rangle$)" if args.sf_use_2pi else r" ($1/\langle k\rangle$)")))

            # Save one file per temperature: sweep, mean L, SE L
            for i, T in enumerate(temps):
                X = np.vstack(L_lists[i])  # (reps, nsamples)
                meanL = X.mean(axis=0)
                seL = X.std(axis=0, ddof=1) / np.sqrt(X.shape[0])

                fname = os.path.join(args.outdir, f"domain_growth_{metric}_N{args.N}_T{T:.6f}_reps{args.reps}.txt")
                np.savetxt(fname, np.column_stack([t, meanL, seL]),
                           header=f"sweep\tMean_{yname}\tSE_{yname}")

                # Save magnetization growth: sweep, mean |m|, SE |m|
                MX = np.vstack(Mabs_lists[i])
                meanM = MX.mean(axis=0)
                seM = MX.std(axis=0, ddof=1) / np.sqrt(MX.shape[0])

                mfile = os.path.join(args.outdir, f"magnetization_growth_N{args.N}_T{T:.6f}_reps{args.reps}.txt")
                np.savetxt(mfile, np.column_stack([t, meanM, seM]),
                           header="sweep\tMean_|m|\tSE_|m|")

            # ===== Plot 1: mean L curves =====
            plt.figure(figsize=(7.2, 5.4))
            for i, T in enumerate(temps):
                X = np.vstack(L_lists[i])
                meanL = X.mean(axis=0)
                plt.loglog(t, meanL, marker='o', markersize=3, linewidth=1.5, label=f"T={T:.3f}")

            plt.xlabel("MC sweeps t")
            plt.ylabel(ylab)
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, title="Temperature")
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.outdir, f"domain_growth_{metric}_mean_N{args.N}_reps{args.reps}_T{args.Tmin}-{args.Tmax}.png"),
                dpi=200, bbox_inches="tight"
            )
            plt.close()

            # ===== Plot 2: mean ± SE (shaded bands) =====
            plt.figure(figsize=(7.2, 5.4))
            for i, T in enumerate(temps):
                X = np.vstack(L_lists[i])
                meanL = X.mean(axis=0)
                seL = X.std(axis=0, ddof=1) / np.sqrt(X.shape[0])

                lower = np.clip(meanL - seL, 1e-12, None)
                upper = meanL + seL

                plt.loglog(t, meanL, marker='o', markersize=3, linewidth=1.5, label=f"T={T:.3f}")
                plt.fill_between(t, lower, upper, alpha=0.18)

            plt.xlabel("MC sweeps t")
            plt.ylabel(ylab + " (mean ± SE)")
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, title="Temperature")
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.outdir, f"domain_growth_{metric}_meanSE_N{args.N}_reps{args.reps}_T{args.Tmin}-{args.Tmax}.png"),
                dpi=200, bbox_inches="tight"
            )
            plt.close()

            # ===== Plot 3: mean |m(t)| curves =====
            plt.figure(figsize=(7.2, 5.0))
            for i, T in enumerate(temps):
                MX = np.vstack(Mabs_lists[i])
                meanM = MX.mean(axis=0)
                plt.semilogx(t, meanM, marker='o', markersize=3, linewidth=1.5, label=f"T={T:.3f}")

            plt.xlabel("MC sweeps t")
            plt.ylabel(r"$\langle |m(t)| \rangle$")
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=10, title="Temperature")
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.outdir, f"magnetization_growth_mean_N{args.N}_reps{args.reps}_T{args.Tmin}-{args.Tmax}.png"),
                dpi=200, bbox_inches="tight"
            )
            plt.close()

            # ===== Plot 4: overlay L(t) and |m(t)| per temperature =====
            for i, T in enumerate(temps):
                X  = np.vstack(L_lists[i])
                MX = np.vstack(Mabs_lists[i])

                meanL = X.mean(axis=0)
                seL   = X.std(axis=0, ddof=1) / np.sqrt(X.shape[0])

                meanM = MX.mean(axis=0)
                seM   = MX.std(axis=0, ddof=1) / np.sqrt(MX.shape[0])

                fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
                ax1.set_xscale("log")
                ax1.plot(t, meanL, marker='o', markersize=3, linewidth=1.5)
                ax1.fill_between(t, np.clip(meanL - seL, 1e-12, None), meanL + seL, alpha=0.18)
                ax1.set_xlabel("MC sweeps t")
                ax1.set_ylabel(ylab)
                ax1.grid(True, which="both", ls="--", alpha=0.3)

                ax2 = ax1.twinx()
                ax2.plot(t, meanM, marker='s', markersize=3, linewidth=1.2)
                ax2.fill_between(t, np.clip(meanM - seM, 0.0, None), np.clip(meanM + seM, 0.0, 1.0), alpha=0.12)
                ax2.set_ylabel(r"$\langle |m(t)| \rangle$")

                plt.title(f"T = {T:.3f}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(args.outdir, f"overlay_L_and_m_{metric}_N{args.N}_T{T:.6f}_reps{args.reps}.png"),
                    dpi=200
                )
                plt.close(fig)

            print(f"Saved domain-growth files + plots in: {args.outdir}")
            print(f"Metric used: {metric}")
            print(f"Each T has: domain_growth_{metric}_N*_T*.txt and magnetization_growth_N*_T*.txt")

    else:
        worker(rank=rank, base_seed=args.seed)

if __name__ == "__main__":
    main()
