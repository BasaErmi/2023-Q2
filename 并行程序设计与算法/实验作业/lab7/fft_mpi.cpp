#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <mpi.h>

using namespace std;

double *local_a;
double *local_b;
double *local_c;
double *local_d;
double *local_w;
double *gather_a;
double *gather_b;
double *gather_c;
double *gather_d;
double *gather_w;
int *c_index;
int *d_index;
int my_rank, comm_sz;
int PER_NUM;

int main();
void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double cpu_time(void);
double ggl(double *ds);
void step(int n, int mj, double a[], double b[], double c[], double d[],
          double w[], double sgn);
void timestamp();

int main()
{
  double ctime;
  double ctime1;
  double ctime2;
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w;
  double *x;
  double *y;
  double *z;
  double z0;
  double z1;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  if (my_rank == 0)
  {
    timestamp();
    cout << "\n";
    cout << "FFT_MPI\n";
    cout << "  C++ version\n";
    cout << "\n";
    cout << "  Demonstrate an implementation of the Fast Fourier Transform\n";
    cout << "  of a complex data vector.\n";

    cout << "\n";
    cout << "  Accuracy check:\n";
    cout << "\n";
    cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
    cout << "\n";
    cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n";
    cout << "\n";
  }

  seed = 331.0;
  n = 1;

  for (ln2 = 1; ln2 <= 20; ln2++)
  {
    n = 2 * n;
    PER_NUM = n / comm_sz;
    if (my_rank == 0)
    {
      w = new double[n];
      x = new double[2 * n];
      y = new double[2 * n];
      z = new double[2 * n];
      gather_a = new double[n];
      gather_b = new double[n];
      gather_c = new double[n];
      gather_d = new double[n];
      gather_w = new double[n];
      c_index = new int[n / 2];
      d_index = new int[n / 2];
    }
    local_a = new double[PER_NUM];
    local_b = new double[PER_NUM];
    local_c = new double[PER_NUM];
    local_d = new double[PER_NUM];
    local_w = new double[PER_NUM];

    first = 1;

    for (icase = 0; icase < 2; icase++)
    {
      if (my_rank == 0)
      {
        if (first)
        {
          for (i = 0; i < 2 * n; i = i + 2)
          {
            z0 = ggl(&seed);
            z1 = ggl(&seed);
            x[i] = z0;
            z[i] = z0;
            x[i + 1] = z1;
            z[i + 1] = z1;
          }
        }
        else
        {
          for (i = 0; i < 2 * n; i = i + 2)
          {
            z0 = 0.0;
            z1 = 0.0;
            x[i] = z0;
            z[i] = z0;
            x[i + 1] = z1;
            z[i + 1] = z1;
          }
        }
        cffti(n, w);
      }

      if (first)
      {
        sgn = +1.0;
        cfft2(n, x, y, w, sgn);
        sgn = -1.0;
        cfft2(n, y, x, w, sgn);

        if (my_rank == 0)
        {
          fnm1 = 1.0 / (double)n;
          error = 0.0;
          for (i = 0; i < 2 * n; i = i + 2)
          {
            error = error + pow(z[i] - fnm1 * x[i], 2) + pow(z[i + 1] - fnm1 * x[i + 1], 2);
          }
          error = sqrt(fnm1 * error);
          if (n >= 16) {
            cout << "  " << setw(12) << n
                 << "  " << setw(8) << nits
                 << "  " << setw(12) << error;
          }
        }
        first = 0;
      }
      else
      {
        ctime1 = MPI_Wtime();
        for (it = 0; it < nits; it++)
        {
          sgn = +1.0;
          cfft2(n, x, y, w, sgn);
          sgn = -1.0;
          cfft2(n, y, x, w, sgn);
        }
        ctime2 = MPI_Wtime();
        if (my_rank == 0)
        {
          ctime = ctime2 - ctime1;

          flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

          mflops = flops / 1.0E+06 / ctime;

          if (n >= 16) {
            cout << "  " << setw(12) << ctime
                 << "  " << setw(12) << ctime / (double)(2 * nits)
                 << "  " << setw(12) << mflops << "\n";
          }
        }
      }
    }

    if (my_rank == 0)
    {
      delete[] w;
      delete[] x;
      delete[] y;
      delete[] z;
      delete[] gather_a;
      delete[] gather_b;
      delete[] gather_c;
      delete[] gather_d;
      delete[] gather_w;
      delete[] c_index;
      delete[] d_index;
    }
    delete[] local_a;
    delete[] local_b;
    delete[] local_c;
    delete[] local_d;
    delete[] local_w;
    if (ln2 % 4 == 0)
    {
      nits = nits / 10;
    }
    if (nits < 1)
    {
      nits = 1;
    }
  }

  if (my_rank == 0)
  {
    cout << "\n";
    cout << "FFT_MPI:\n";
    cout << "  Normal end of execution.\n";
    cout << "\n";
    timestamp();
  }

  MPI_Finalize();
  return 0;
}

void ccopy(int n, double x[], double y[])
{
  int i;

  for (i = 0; i < n; i++)
  {
    y[i * 2 + 0] = x[i * 2 + 0];
    y[i * 2 + 1] = x[i * 2 + 1];
  }
  return;
}

void cfft2(int n, double x[], double y[], double w[], double sgn)
{
  int j;
  int m;
  int mj;
  int tgle;

  m = (int)(log((double)n) / log(1.99));
  mj = 1;

  tgle = 1;
  step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

  if (n == 2)
  {
    return;
  }

  for (j = 0; j < m - 2; j++)
  {
    mj = mj * 2;
    if (tgle)
    {
      step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
      tgle = 0;
    }
    else
    {
      step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
      tgle = 1;
    }
  }

  if (my_rank == 0)
  {
    if (tgle)
    {
      ccopy(n, y, x);
    }
  }

  mj = n / 2;
  step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

  return;
}

void cffti(int n, double w[])
{
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ((double)n);

  for (i = 0; i < n2; i++)
  {
    arg = aw * ((double)i);
    w[i * 2 + 0] = cos(arg);
    w[i * 2 + 1] = sin(arg);
  }
  return;
}

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

double ggl(double *seed)
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = *seed;
  t = fmod(16807.0 * t, d2);
  *seed = t;
  value = (t - 1.0) / (d2 - 1.0);

  return value;
}

void step(int n, int mj, double a[], double b[], double c[],
          double d[], double w[], double sgn)
{
  if (my_rank == 0)
  {
    int j;
    int ja;
    int jb;
    int jc;
    int jd;
    int jw;
    int k;
    int lj;
    int mj2;
    double wjw[2];

    mj2 = 2 * mj;
    lj = n / mj2;

    int cnt = 0;
    for (j = 0; j < lj; j++)
    {
      jw = j * mj;
      ja = jw;
      jb = ja;
      jc = j * mj2;
      jd = jc;

      wjw[0] = w[jw * 2 + 0];
      wjw[1] = w[jw * 2 + 1];

      if (sgn < 0.0)
      {
        wjw[1] = -wjw[1];
      }

      for (k = 0; k < mj; k++)
      {
        gather_a[cnt * 2 + 0] = a[(ja + k) * 2 + 0];
        gather_a[cnt * 2 + 1] = a[(ja + k) * 2 + 1];
        gather_b[cnt * 2 + 0] = b[(jb + k) * 2 + 0];
        gather_b[cnt * 2 + 1] = b[(jb + k) * 2 + 1];
        gather_w[cnt * 2 + 0] = wjw[0];
        gather_w[cnt * 2 + 1] = wjw[1];
        c_index[cnt] = jc + k;
        d_index[cnt] = jd + k;
        cnt++;
      }
    }
  }

  MPI_Scatter(gather_a, PER_NUM, MPI_DOUBLE, local_a, PER_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(gather_b, PER_NUM, MPI_DOUBLE, local_b, PER_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(gather_w, PER_NUM, MPI_DOUBLE, local_w, PER_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int num_units = PER_NUM / 2 ;
  for (int i = 0; i < num_units; i++)
  {
    local_c[i * 2 + 0] = local_a[i * 2 + 0] + local_b[i * 2 + 0];
    local_c[i * 2 + 1] = local_a[i * 2 + 1] + local_b[i * 2 + 1];
    double ambr = local_a[i * 2 + 0] - local_b[i * 2 + 0];
    double ambu = local_a[i * 2 + 1] - local_b[i * 2 + 1];
    local_d[i * 2 + 0] = local_w[i * 2 + 0] * ambr - local_w[i * 2 + 1] * ambu;
    local_d[i * 2 + 1] = local_w[i * 2 + 1] * ambr + local_w[i * 2 + 0] * ambu;
  }

  MPI_Gather(local_c, PER_NUM, MPI_DOUBLE, gather_c, PER_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(local_d, PER_NUM, MPI_DOUBLE, gather_d, PER_NUM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (my_rank == 0)
  {
    for (int i = 0; i < n / 2; i++)
    {
      c[c_index[i] * 2 + 0] = gather_c[i * 2 + 0];
      c[c_index[i] * 2 + 1] = gather_c[i * 2 + 1];
      d[d_index[i] * 2 + 0] = gather_d[i * 2 + 0];
      d[d_index[i] * 2 + 1] = gather_d[i * 2 + 1];
    }
  }


  return;
}

void timestamp()
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  return;
#undef TIME_SIZE
}