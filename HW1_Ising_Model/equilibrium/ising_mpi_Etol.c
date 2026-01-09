// ising_mpi.c
// Build: mpicc -O3 -march=native -ffast-math ising_mpi.c -o ising_mpi -lm
// Run:   mpirun -np 8 ./ising_mpi --N 64 --Tmin 1.5 --Tmax 3.0 --steps 20 --reps 10

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

enum { TAG_TASK = 1, TAG_RESULT = 2, TAG_STOP = 9 };

typedef struct {
  int N;
  double Tmin, Tmax;
  int steps;
  int reps;
  double tol;
  int window;
  int max_sweeps;
  int meas_sweeps;
  uint64_t seed;
  char outpath[512];
} Params;

/* ---------------- RNG: xorshift64* ---------------- */
static inline uint64_t xorshift64star(uint64_t *s){
  uint64_t x = *s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *s = x;
  return x * 2685821657736338717ULL;
}
static inline double rng_u01(uint64_t *s){
  // 53-bit mantissa uniform in [0,1)
  return (xorshift64star(s) >> 11) * (1.0/9007199254740992.0);
}
static inline int rng_int(uint64_t *s, int n){
  // simple modulo; OK for MC here. For strict uniformity use rejection.
  return (int)(xorshift64star(s) % (uint64_t)n);
}

/* ---------------- Lattice helpers ---------------- */
static inline int idx(int x, int y, int N){ return x*N + y; }

static void init_lattice(int *lat, int N, uint64_t *rng){
  int NN = N*N;
  for(int k=0;k<NN;k++){
    lat[k] = (rng_u01(rng) < 0.5) ? -1 : +1;
  }
}

static double magnetization(const int *lat, int N){
  long sum = 0;
  int NN = N*N;
  for(int k=0;k<NN;k++) sum += lat[k];
  return (double)sum / (double)NN;
}

// energy per spin: count right and down neighbors only (same convention as your Python)
static double energy_per_spin(const int *lat, int N){
  long E = 0;
  for(int x=0;x<N;x++){
    int xp = (x+1)%N;
    for(int y=0;y<N;y++){
      int yp = (y+1)%N;
      int s  = lat[idx(x,y,N)];
      E -= s * (lat[idx(xp,y,N)] + lat[idx(x,yp,N)]);
    }
  }
  return (double)E / (double)(N*N);
}

static inline void metropolis_one_flip(int *lat, int N, double T, uint64_t *rng){
  int x = rng_int(rng, N);
  int y = rng_int(rng, N);

  int xm = (x-1+N)%N, xp=(x+1)%N;
  int ym = (y-1+N)%N, yp=(y+1)%N;

  int s  = lat[idx(x,y,N)];
  int nn = lat[idx(xp,y,N)] + lat[idx(xm,y,N)] + lat[idx(x,yp,N)] + lat[idx(x,ym,N)];
  int dE = 2 * s * nn; // ΔE for flipping s -> -s

  if(dE <= 0){
    lat[idx(x,y,N)] = -s;
  } else {
    double r = rng_u01(rng);
    if(r < exp(-(double)dE / T)){
      lat[idx(x,y,N)] = -s;
    }
  }
}

static void do_sweep(int *lat, int N, double T, uint64_t *rng){
  int NN = N*N;
  for(int k=0;k<NN;k++){
    metropolis_one_flip(lat, N, T, rng);
  }
}

/* ---------------- Simulation ----------------
   Includes:
   - size scaling (Metropolis z ≈ 2)
   - convergence check on BOTH |M| and E (energy criterion)
*/
static void simulate_ising(
  int N, double T,
  double tol, int window,
  int max_sweeps, int meas_sweeps,
  uint64_t *rng,
  double *out_M_mean_abs, double *out_E_mean
){
  int *lat = (int*)malloc((size_t)N*N*sizeof(int));
  if(!lat){ fprintf(stderr,"malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

  init_lattice(lat, N, rng);

  /* -------- size scaling -------- */
  const int    N_ref = 64;
  const double z     = 2.0;                 // Metropolis dynamic exponent (practical)
  double factor = pow((double)N / (double)N_ref, z);
  if (factor < 1.0) factor = 1.0;

  int t_eq_min = (int)llround(0.5 * (double)max_sweeps * factor);
  int t_eq_max = (int)llround(1.0 * (double)max_sweeps * factor);
  int t_meas   = (int)llround(1.0 * (double)meas_sweeps * factor);

  // window needs to grow too, but don't let it explode (memory/time)
  int win = (int)llround((double)window * sqrt(factor));
  if (win < window) win = window;

  // records for equilibration diagnostics (scaled)
  double *Mrec = (double*)malloc((size_t)(t_eq_max+1)*sizeof(double));
  double *Erec = (double*)malloc((size_t)(t_eq_max+1)*sizeof(double));
  if(!Mrec || !Erec){ fprintf(stderr,"malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

  int sweep = 0;
  int eq = 0;

  while(sweep < t_eq_max && !eq){
    do_sweep(lat, N, T, rng);
    sweep++;

    double m = magnetization(lat, N);
    double e = energy_per_spin(lat, N);

    Mrec[sweep-1] = fabs(m);
    Erec[sweep-1] = e;

    // don’t even try criteria before minimum burn-in and 2 windows
    if(sweep < t_eq_min || sweep < 2*win) continue;

    // window means for |M| and E
    double m_recent=0.0, m_prev=0.0;
    double e_recent=0.0, e_prev=0.0;
    for(int k=0;k<win;k++){
      m_recent += Mrec[sweep-1-k];
      m_prev   += Mrec[sweep-1-win-k];

      e_recent += Erec[sweep-1-k];
      e_prev   += Erec[sweep-1-win-k];
    }
    m_recent /= (double)win;  m_prev /= (double)win;
    e_recent /= (double)win;  e_prev /= (double)win;

    double deltaM = fabs(m_recent - m_prev) / (fabs(m_prev) + 1e-12);
    double deltaE = fabs(e_recent - e_prev) / (fabs(e_prev) + 1e-12);

    // Energy criterion: require BOTH to be stable
    if(deltaM < tol && deltaE < tol) eq = 1;
  }

  // measurement phase (scaled length)
  double Msum=0.0, Esum=0.0;
  for(int t=0;t<t_meas;t++){
    do_sweep(lat, N, T, rng);
    double m = magnetization(lat, N);
    Msum += fabs(m);
    Esum += energy_per_spin(lat, N);
  }

  *out_M_mean_abs = Msum / (double)t_meas;
  *out_E_mean     = Esum / (double)t_meas;

  free(Erec);
  free(Mrec);
  free(lat);
}

/* ---------------- CLI parsing (minimal) ---------------- */
static void set_defaults(Params *p){
  p->N = 64;
  p->Tmin = 1.5;
  p->Tmax = 3.0;
  p->steps = 20;
  p->reps = 10;
  p->tol = 0.01;
  p->window = 1000;
  p->max_sweeps = 50000;
  p->meas_sweeps = 5000;
  p->seed = 12345ULL;
  snprintf(p->outpath, sizeof(p->outpath), "ising_data_mpi.tsv");
}

static void parse_args(int argc, char **argv, Params *p){
  set_defaults(p);
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--N") && i+1<argc) p->N = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--Tmin") && i+1<argc) p->Tmin = atof(argv[++i]);
    else if(!strcmp(argv[i],"--Tmax") && i+1<argc) p->Tmax = atof(argv[++i]);
    else if(!strcmp(argv[i],"--steps") && i+1<argc) p->steps = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--reps") && i+1<argc) p->reps = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--tol") && i+1<argc) p->tol = atof(argv[++i]);
    else if(!strcmp(argv[i],"--window") && i+1<argc) p->window = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--max_sweeps") && i+1<argc) p->max_sweeps = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--meas_sweeps") && i+1<argc) p->meas_sweeps = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--seed") && i+1<argc) p->seed = (uint64_t)strtoull(argv[++i], NULL, 10);
    else if(!strcmp(argv[i],"--out") && i+1<argc) snprintf(p->outpath,sizeof(p->outpath),"%s",argv[++i]);
    else if(!strcmp(argv[i],"--help")){
      // minimal help stub (kept as-is)
      if(0) printf("help\n");
    }
  }
}

/* ---------------- Master/Worker ---------------- */
static void worker_loop(const Params *p, int rank){
  MPI_Status st;
  while(1){
    int task[3]; // i, rep, steps (steps included for safety)
    double T;
    MPI_Recv(task, 3, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    if(st.MPI_TAG == TAG_STOP) break;
    if(st.MPI_TAG != TAG_TASK) continue;

    MPI_Recv(&T, 1, MPI_DOUBLE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int i = task[0];
    int rep = task[1];

    // deterministic per-(i,rep) seed independent of rank
    uint64_t s = p->seed;
    s += 1000003ULL * (uint64_t)rep;
    s += 9721ULL    * (uint64_t)i;
    if(s == 0) s = 0x9e3779b97f4a7c15ULL;

    double Mmean, Emean;
    simulate_ising(p->N, T, p->tol, p->window, p->max_sweeps, p->meas_sweeps, &s, &Mmean, &Emean);

    // result: i, rep, T, M, E
    int outi[2] = { i, rep };
    double outd[3] = { T, Mmean, Emean };
    MPI_Send(outi, 2, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
    MPI_Send(outd, 3, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
  }
  (void)rank;
}

static double *linspace(double a, double b, int n){
  double *x = (double*)malloc((size_t)n*sizeof(double));
  if(!x){ fprintf(stderr,"malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
  if(n<=1){ x[0]=a; return x; }
  for(int i=0;i<n;i++){
    x[i] = a + (b-a) * ((double)i / (double)(n-1));
  }
  return x;
}

static void master_loop(const Params *p, int size){
  int Tn = p->steps;
  double *temps = linspace(p->Tmin, p->Tmax, Tn);

  // storage: per temperature, arrays of replicate values
  double *Mvals = (double*)calloc((size_t)Tn * (size_t)p->reps, sizeof(double));
  double *Evals = (double*)calloc((size_t)Tn * (size_t)p->reps, sizeof(double));
  int    *count = (int*)calloc((size_t)Tn, sizeof(int));
  if(!Mvals || !Evals || !count){ fprintf(stderr,"alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

  int total_tasks = Tn * p->reps;
  int next_task = 0;
  int inflight = 0;

  // send initial tasks
  for(int w=1; w<size; w++){
    if(next_task >= total_tasks) break;
    int i = next_task / p->reps;
    int rep = next_task % p->reps;
    int task[3] = { i, rep, Tn };
    double T = temps[i];
    MPI_Send(task, 3, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
    MPI_Send(&T, 1, MPI_DOUBLE, w, TAG_TASK, MPI_COMM_WORLD);
    next_task++; inflight++;
  }

  int done = 0;
  MPI_Status st;
  while(inflight > 0){
    int outi[2];
    double outd[3];
    MPI_Recv(outi, 2, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &st);
    MPI_Recv(outd, 3, MPI_DOUBLE, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int i = outi[0];
    int rep = outi[1];
    double T = outd[0], M = outd[1], E = outd[2];

    Mvals[i*p->reps + rep] = M;
    Evals[i*p->reps + rep] = E;
    count[i]++;

    inflight--;
    done++;

    printf("[%d/%d] T=%.3f rep=%d -> %d/%d completed for T\n",
           done, total_tasks, T, rep, count[i], p->reps);
    fflush(stdout);

    if(count[i] == p->reps){
      printf("--> Temperature %.3f finished all %d replicates\n", T, p->reps);
      fflush(stdout);
    }

    // feed next task to the worker that just finished
    if(next_task < total_tasks){
      int i2 = next_task / p->reps;
      int rep2 = next_task % p->reps;
      int task[3] = { i2, rep2, Tn };
      double T2 = temps[i2];
      MPI_Send(task, 3, MPI_INT, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
      MPI_Send(&T2, 1, MPI_DOUBLE, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
      next_task++; inflight++;
    }
  }

  // stop workers
  for(int w=1; w<size; w++){
    int dummy[3] = {0,0,0};
    MPI_Send(dummy, 3, MPI_INT, w, TAG_STOP, MPI_COMM_WORLD);
  }

  // aggregate means + SE
  FILE *f = fopen(p->outpath, "w");
  if(!f){
    fprintf(stderr,"Could not open output file %s\n", p->outpath);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  fprintf(f, "Temperature\tMean_|M|\tSE_|M|\tMean_E_per_Spin\tSE_E\n");

  for(int i=0;i<Tn;i++){
    double msum=0.0, esum=0.0;
    for(int r=0;r<p->reps;r++){
      msum += Mvals[i*p->reps + r];
      esum += Evals[i*p->reps + r];
    }
    double mmean = msum / (double)p->reps;
    double emean = esum / (double)p->reps;

    // sample std -> SE
    double mvar=0.0, evar=0.0;
    for(int r=0;r<p->reps;r++){
      double dm = Mvals[i*p->reps + r] - mmean;
      double de = Evals[i*p->reps + r] - emean;
      mvar += dm*dm;
      evar += de*de;
    }
    double mstd = (p->reps>1) ? sqrt(mvar / (double)(p->reps-1)) : 0.0;
    double estd = (p->reps>1) ? sqrt(evar / (double)(p->reps-1)) : 0.0;
    double mse  = mstd / sqrt((double)p->reps);
    double ese  = estd / sqrt((double)p->reps);

    fprintf(f, "%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", temps[i], mmean, mse, emean, ese);
    printf("T=%.3f -> mean<|M|>=%.6f, mean<E>/spin=%.6f\n", temps[i], mmean, emean);
  }
  fclose(f);

  free(count);
  free(Mvals);
  free(Evals);
  free(temps);
}

int main(int argc, char **argv){
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Params p;
  parse_args(argc, argv, &p);

  if(size < 2 && rank==0){
    fprintf(stderr,"Need at least 2 MPI ranks (1 master + workers)\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if(rank == 0) master_loop(&p, size);
  else worker_loop(&p, rank);

  MPI_Finalize();
  return 0;
}
