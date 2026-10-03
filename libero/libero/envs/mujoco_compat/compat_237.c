// Copyright 2016 Svetoslav Kolev
// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. A copy of the License is in LICENSE next to this file.
//
// Modified from mujoco 2.3.7 (src/engine/engine_collision_box.c, engine_util_blas.c,
// engine_support.c) for use with mujoco >= 3.9; the changes are described below.
//
// mujoco 2.3.7 behaviour for mujoco >= 3.9, used by mujoco_compat/__init__.py. Two parts:
//
// 1. compat_BoxBox_old: mujoco 2.3.7 mjc_BoxBox ported to the mjfCollision / mjPreContact interface.
//    Source: google-deepmind/mujoco tag 2.3.7, src/engine/engine_collision_box.c lines 596-1332
//    (Copyright 2016 Svetoslav Kolev, Apache License 2.0). The function body is verbatim except:
//    mjContact -> oldcon_t, mju_* -> old_* (static copies of the 2.3.7 helpers below), static linkage.
//    The wrapper reproduces the 2.3.7 mj_collideGeoms post-processing for box-box (exact-duplicate
//    position removal). Why: 3.4.0 (commit 88383684) changed the face-path depth from points[i][2] to
//    2*points[i][2], and 3.12.0 rewrote the collider.
// 2. compat_normalizeQuat_237: the in-place, unconditional quaternion normalization that 2.3.7
//    mj_kinematics does first (engine_core_smooth.c:49-55, mj_normalizeQuat engine_support.c:1426-1433,
//    mju_normalize4 engine_util_blas.c:216-232). 3.14 normalizes a copy and skips |norm-1| <= 1e-15,
//    so an init quaternion with |q|-1 = 2.2e-16 gives a different geom_xmat.
//
// Build: gcc -O2 -ffp-contract=off -fPIC -shared (FMA contraction breaks bit parity with 2.3.7).
#include <math.h>
#include <string.h>
#include <stdatomic.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjdata.h>

typedef struct { mjtNum dist; mjtNum pos[3]; mjtNum frame[9]; int bad; } oldcon_t;

// ---- 2.3.7 engine_util_blas.c helpers (n=3 paths only are reached; copied verbatim) ----
static inline mjtNum old_abs(mjtNum x) { return fabs(x); }
static inline void old_add3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3]) {
  res[0] = vec1[0] + vec2[0]; res[1] = vec1[1] + vec2[1]; res[2] = vec1[2] + vec2[2]; }
static inline void old_addToScl3(mjtNum res[3], const mjtNum vec[3], mjtNum scl) {
  res[0] += vec[0] * scl; res[1] += vec[1] * scl; res[2] += vec[2] * scl; }
static inline void old_addToScl(mjtNum* res, const mjtNum* vec, mjtNum scl, int n) {
  for (int i = 0; i < n; i++) res[i] += vec[i]*scl; }
static inline void old_copy(mjtNum* res, const mjtNum* vec, int n) { if (n > 0) memcpy(res, vec, n*sizeof(mjtNum)); }
static inline void old_copy3(mjtNum res[3], const mjtNum data[3]) { res[0] = data[0]; res[1] = data[1]; res[2] = data[2]; }
static inline mjtNum old_dot3(const mjtNum vec1[3], const mjtNum vec2[3]) {
  return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]; }
static inline mjtNum old_dot(const mjtNum* vec1, const mjtNum* vec2, int n) {
  // 2.3.7 mju_dot for n < 4: res = 0; res += (a0*b0 + a1*b1 + a2*b2)
  mjtNum res = 0; int i = 0; int n_i = n - i;
  if (n >= 4) { for (int k = 0; k < n; k++) res += vec1[k]*vec2[k]; return res; }  // not reached (n==3)
  if (n_i == 3) res += vec1[i]*vec2[i] + vec1[i+1]*vec2[i+1] + vec1[i+2]*vec2[i+2];
  else if (n_i == 2) res += vec1[i]*vec2[i] + vec1[i+1]*vec2[i+1];
  else if (n_i == 1) res += vec1[i]*vec2[i];
  return res; }
static inline void old_zero(mjtNum* res, int n) { if (n > 0) memset(res, 0, n*sizeof(mjtNum)); }
static inline void old_zero3(mjtNum res[3]) { res[0] = 0; res[1] = 0; res[2] = 0; }
static inline void old_mulMatMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2, int r1, int c1, int c2) {
  mjtNum tmp; old_zero(res, r1*c2);
  for (int i=0; i < r1; i++) for (int k=0; k < c1; k++) if ((tmp = mat1[i*c1+k])) old_addToScl(res+i*c2, mat2+k*c2, tmp, c2); }
static inline void old_mulMatMatT(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2, int r1, int c1, int r2) {
  for (int i=0; i < r1; i++) for (int j=0; j < r2; j++) res[i*r2+j] = old_dot(mat1+i*c1, mat2+j*c1, c1); }
static inline void old_mulMatTMat(mjtNum* res, const mjtNum* mat1, const mjtNum* mat2, int r1, int c1, int c2) {
  mjtNum tmp; old_zero(res, c1*c2);
  for (int i=0; i < r1; i++) for (int j=0; j < c1; j++) if ((tmp = mat1[i*c1+j])) old_addToScl(res+j*c2, mat2+i*c2, tmp, c2); }
static inline mjtNum old_normalize3(mjtNum vec[3]) {
  mjtNum norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  if (norm < mjMINVAL) { vec[0] = 1; vec[1] = 0; vec[2] = 0; }
  else { mjtNum normInv = 1/norm; vec[0] *= normInv; vec[1] *= normInv; vec[2] *= normInv; }
  return norm; }
static inline void old_rotVecMat(mjtNum res[3], const mjtNum vec[3], const mjtNum mat[9]) {
  mjtNum tmp[3] = { mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2],
                    mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2],
                    mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2] };
  res[0] = tmp[0]; res[1] = tmp[1]; res[2] = tmp[2]; }
static inline void old_rotVecMatT(mjtNum res[3], const mjtNum vec[3], const mjtNum mat[9]) {
  mjtNum tmp[3] = { mat[0]*vec[0] + mat[3]*vec[1] + mat[6]*vec[2],
                    mat[1]*vec[0] + mat[4]*vec[1] + mat[7]*vec[2],
                    mat[2]*vec[0] + mat[5]*vec[1] + mat[8]*vec[2] };
  res[0] = tmp[0]; res[1] = tmp[1]; res[2] = tmp[2]; }
static inline void old_scl3(mjtNum res[3], const mjtNum vec[3], mjtNum scl) {
  res[0] = vec[0] * scl; res[1] = vec[1] * scl; res[2] = vec[2] * scl; }
static inline void old_sub3(mjtNum res[3], const mjtNum vec1[3], const mjtNum vec2[3]) {
  res[0] = vec1[0] - vec2[0]; res[1] = vec1[1] - vec2[1]; res[2] = vec1[2] - vec2[2]; }
static inline void old_transpose(mjtNum* res, const mjtNum* mat, int nr, int nc) {
  for (int i=0; i < nr; i++) for (int j=0; j < nc; j++) res[j*nr+i] = mat[i*nc+j]; }

// ---- verbatim 2.3.7 mjc_BoxBox (renamed) ----

static int old_BoxBox(const mjModel* M, const mjData* D, oldcon_t* con, int g1, int g2, mjtNum margin)
{
  const mjtNum* pos1 = D->geom_xpos + 3 * g1;
  const mjtNum* mat1 = D->geom_xmat + 9 * g1;
  const mjtNum* size1 = M->geom_size + 3 * g1;
  const mjtNum* pos2 = D->geom_xpos + 3 * g2;
  const mjtNum* mat2 = D->geom_xmat + 9 * g2;
  const mjtNum* size2 = M->geom_size + 3 * g2;

  mjtNum pos12[3], pos21[3], rot[9], rott[9], rotabs[9], rottabs[9], tmp1[3], tmp2[3], plen1[3],
         plen2[3];
  mjtNum rotmore[9], p[3], r[9], s[3], ss[3], lp[3], rt[9], points[mjMAXCONPAIR][3],
         depth[mjMAXCONPAIR], pts[6][3], ppts2[4][2], pu[4][3], axi[3][3];
  mjtNum linesu[4][6], lines[4][6], clnorm[3], rnorm[3];
  mjtNum penetration, c1, c2, c3, a, b, c, d, lx, ly, hz, l, x, y, u, v, llx, lly, innorm, margin2;

  int i0, i1, i2;
  mjtNum f0, f1, f2;

  int i, j, q, code, q1, q2, clcorner, n, m, k;
  int cle1, cle2, in, ax1, ax2, pax1, pax2, clface, nl, nf;

  n = 0;
  code = -1;
  margin2 = margin * margin;

  old_sub3(tmp1, pos2, pos1);
  old_rotVecMatT(pos21, tmp1, mat1);

  old_sub3(tmp1, pos1, pos2);
  old_rotVecMatT(pos12, tmp1, mat2);

  old_mulMatTMat(rot, mat1, mat2, 3, 3, 3);
  old_transpose(rott, rot, 3, 3);

  for (i = 0; i < 9; i++)
    rotabs[i] = fabs(rot[i]);
  for (i = 0; i < 9; i++)
    rottabs[i] = fabs(rott[i]);

  old_rotVecMat(plen2, size2, rotabs);
  old_rotVecMatT(plen1, size1, rotabs);

  for (i = 0, penetration = margin; i < 3; i++)
    penetration += size1[i] * 3 + size2[i] * 3;

  for (i = 0; i < 3; i++) {
    c1 = -fabs(pos21[i]) + size1[i] + plen2[i];
    c2 = -fabs(pos12[i]) + size2[i] + plen1[i];

    if (c1 < -margin || c2 < -margin)
      return 0;

    if (c1 < penetration) {
      penetration = c1;
      code = i + 3 * (pos21[i] < 0) + 0;
    }
    if (c2 < penetration) {
      penetration = c2;
      code = i + 3 * (pos12[i] < 0) + 6;
    }

    // printf("%24.16e %24.16e %d         %24.16e %d \n",c1,c2,i,penetration,code);
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      old_zero3(tmp2);
      if (i == 0) {
        tmp2[1] = -rott[3 * j + 2];
        tmp2[2] = +rott[3 * j + 1];
      } else if (i == 1) {
        tmp2[0] = +rott[3 * j + 2];
        tmp2[2] = -rott[3 * j + 0];
      } else if (i == 2) {
        tmp2[0] = -rott[3 * j + 1];
        tmp2[1] = +rott[3 * j + 0];
      }

      c1 = old_normalize3(tmp2);


      if (c1 < mjMINVAL)
        continue;

      c2 = old_dot3(pos21, tmp2);

      c3 = 0;

      for (k = 0; k < 3; k++)
        if (k != i)
          c3 += size1[k] * fabs(tmp2[k]);
      for (k = 0; k < 3; k++)
        if (k != j)
          c3 += size2[k] * rotabs[3 * i + 3 - k - j] / c1;

      c3 -= fabs(c2);

      if (c3 < -margin)
        return 0;



      if (c3 < penetration * (1 - 1e-12))
      {
        penetration = c3;
        for (k = cle1 = 0; k < 3; k++)
          if (k != i)
            if ((tmp2[k] > 0) ^ (c2 < 0))
              cle1 += 1 << k;
        for (k = cle2 = 0; k < 3; k++)
          if (k != j)
            if ((rot[3 * i + 3 - k - j] > 0) ^ (c2 < 0) ^ ((k - j + 3) % 3 == 1))
              cle2 += 1 << k;

        code = 12 + i * 3 + j;
        old_copy3(clnorm, tmp2);
        in = c2 < 0;
      }

      // printf("%24.16e %d      %24.16e %d\n",c3,12+i*3+j,penetration,code);
    }
  }


  // return 0;


  // printf("%d\n",code);

  if (code == -1)
    return 0;  // shouldn't happen

  if (code >= 12)
    goto edgeedge;


  q1 = code % 6;
  q2 = code / 6;

  // printf("%d %d\n",q1,q2);

  old_zero(rotmore, 9);
  if (q1 == 0)
    rotmore[2] = -1, rotmore[4] = +1, rotmore[6] = +1;
  else if (q1 == 1)
    rotmore[0] = +1, rotmore[5] = -1, rotmore[7] = +1;
  else if (q1 == 2)
    rotmore[0] = +1, rotmore[4] = +1, rotmore[8] = +1;
  else if (q1 == 3)
    rotmore[2] = +1, rotmore[4] = +1, rotmore[6] = -1;
  else if (q1 == 4)
    rotmore[0] = +1, rotmore[5] = +1, rotmore[7] = -1;
  else if (q1 == 5)
    rotmore[0] = -1, rotmore[4] = +1, rotmore[8] = -1;

  i0 = 0;
  i1 = 1;
  i2 = 2;
  f0 = f1 = f2 = 1;

  if (q1 == 0) {
    i0 = 2;
    f0 = -1;
    i2 = 0;
  } else if (q1 == 1) {
    i1 = 2;
    f1 = -1;
    i2 = 1;
  } else if (q1 == 2) {
  } else if (q1 == 3) {
    i0 = 2;
    i2 = 0;
    f2 = -1;
  } else if (q1 == 4) {
    i1 = 2;
    i2 = 1;
    f2 = -1;
  } else if (q1 == 5) {
    f0 = -1;
    f2 = -1;
  }


#define rotaxis(vecres, vecin) \
{                              \
  vecres[0]=vecin[i0]*f0;      \
  vecres[1]=vecin[i1]*f1;      \
  vecres[2]=vecin[i2]*f2;      \
}
#define rotmatx(matres, matin)        \
{                                     \
  old_scl3(matres+0, matin+i0*3, f0); \
  old_scl3(matres+3, matin+i1*3, f1); \
  old_scl3(matres+6, matin+i2*3, f2); \
}

  if (q2) {
    old_mulMatMatT(r, rotmore, rot, 3, 3, 3);

    // old_rotVecMat(p,pos12,rotmore);
    // old_rotVecMat(tmp1,size2,rotmore);

    rotaxis(p, pos12);
    rotaxis(tmp1, size2);

    old_copy3(s, size1);
  } else {
    // old_mulMatMat(r,rotmore,rot,3,3,3);

    rotmatx(r, rot);

    // old_rotVecMat(p,pos21,rotmore);
    // old_rotVecMat(tmp1,size1,rotmore);

    rotaxis(p, pos21);
    rotaxis(tmp1, size1);

    old_copy3(s, size2);
  }

  old_transpose(rt, r, 3, 3);

  for (i = 0; i < 3; i++)
    ss[i] = old_abs(tmp1[i]);

  lx = ss[0];
  ly = ss[1];
  hz = ss[2];
  p[2] -= hz;

  old_copy3(lp, p);

  for (clcorner = 0, i = 0; i < 3; i++)
    if (r[6 + i] < 0)
      clcorner += 1 << i;

  old_addToScl3(lp, rt + 0, s[0] * ((clcorner & 1) ? 1 : -1));
  old_addToScl3(lp, rt + 3, s[1] * ((clcorner & 2) ? 1 : -1));
  old_addToScl3(lp, rt + 6, s[2] * ((clcorner & 4) ? 1 : -1));

  m = k = 0;
  old_copy3(pts[m++], lp);

  for (i = 0; i < 3; i++)
    if (fabs(r[6 + i]) < 0.5)
      old_scl3(pts[m++], rt + 3 * i, s[i] * ((clcorner & (1 << i)) ? -2 : 2));

  old_add3(pts[3], pts[0], pts[1]);
  old_add3(pts[4], pts[0], pts[2]);
  old_add3(pts[5], pts[3], pts[2]);

  if (m > 1)
  {
    old_copy3(lines[k] + 0, pts[0]);
    old_copy3(lines[k++] + 3, pts[1]);
  }
  if (m > 2)
  {
    old_copy3(lines[k] + 0, pts[0]);
    old_copy3(lines[k++] + 3, pts[2]);
    old_copy3(lines[k] + 0, pts[3]);
    old_copy3(lines[k++] + 3, pts[2]);
    old_copy3(lines[k] + 0, pts[4]);
    old_copy3(lines[k++] + 3, pts[1]);
  }

  for (i = 0; i < k; i++) {
    for (q = 0; q < 2; q++) {
      a = lines[i][0 + q];
      b = lines[i][3 + q];
      c = lines[i][1 - q];
      d = lines[i][4 - q];

      if (fabs(b) > mjMINVAL) {
        for (j = -1; j <= 1; j += 2) {
          l = ss[q] * j;
          c1 = (l - a) * (1 / b);
          if (c1 < 0 || c1 > 1)
            continue;
          c2 = c + d * c1;
          if (fabs(c2) > ss[1 - q])
            continue;

          old_copy3(points[n], lines[i]);
          old_addToScl3(points[n++], lines[i] + 3, c1);
        }
      }
    }
  }


  a = pts[1][0];
  b = pts[2][0];
  c = pts[1][1];
  d = pts[2][1];
  c1 = a * d - b * c;


  if (m > 2) {
    for (i = 0; i < 4; i++) {
      llx = i / 2 ? lx : -lx;
      lly = i % 2 ? ly : -ly;

      x = llx - pts[0][0];
      y = lly - pts[0][1];

      u = (x * d - y * b) * (1 / c1);
      v = (y * a - x * c) * (1 / c1);
      if (u <= 0 || v <= 0 || u >= 1 || v >= 1)
        continue;

      points[n][0] = llx;
      points[n][1] = lly;
      points[n][2] = (pts[0][2] + u * pts[1][2] + v * pts[2][2]);
      n++;
    }
  }

  for (i = 0; i < (1 << (m - 1)); i++) {
    old_copy3(tmp1, pts[i == 0 ? 0 : i + 2]);


    if (i)
      if (tmp1[0] <= -lx || tmp1[0] >= lx)
        continue;
    if (i)
      if (tmp1[1] <= -ly || tmp1[1] >= ly)
        continue;

    old_copy3(points[n++], tmp1);
  }


  m = n;
  n = 0;

  for (i = 0; i < m; i++)
  {
    if (points[i][2] > margin)
      continue;
    old_copy3(points[n], points[i]);

    depth[n] = points[n][2];
    points[n][2] *= 0.5;

    n++;
  }


  old_mulMatMatT(r, q2 ? mat2 : mat1, rotmore, 3, 3, 3);
  old_copy3(p, q2 ? pos2 : pos1);

  tmp2[0] = (q2 ? -1 : 1) * r[2];
  tmp2[1] = (q2 ? -1 : 1) * r[5];
  tmp2[2] = (q2 ? -1 : 1) * r[8];

  old_copy3(con[0].frame, tmp2);
  old_zero3(con[0].frame + 3);




  for (i = 0; i < n; i++)
  {
    con[i].dist = points[i][2];
    points[i][2] += hz;

    old_rotVecMat(tmp2, points[i], r);
    old_add3(con[i].pos, tmp2, p);

    if (i)
      old_copy(con[i].frame, con[0].frame, 6);
  }


  // printf("Path1:  %d\n",n);


  return n;

edgeedge:


  code -= 12;

  q1 = code / 3;
  q2 = code % 3;



  if (q2 == 0)
    ax1 = 1, ax2 = 2;
  if (q2 == 1)
    ax1 = 0, ax2 = 2;
  if (q2 == 2)
    ax1 = 1, ax2 = 0;
  if (q1 == 0)
    pax1 = 1, pax2 = 2;
  if (q1 == 1)
    pax1 = 0, pax2 = 2;
  if (q1 == 2)
    pax1 = 1, pax2 = 0;

  // printf("%lf %lf   %lf %lf\n",rot[ 3*q1+ ax1],rot [3*q1+ ax2],rott[3*q2+pax1],rott[3*q2+pax2]);
  // printf("%lf %lf\n",old_dot3(clnorm,rott+3*ax1),old_dot3(clnorm,rott+3*ax2));

  if (rotabs [3 * q1 + ax1] < rotabs [3 * q1 + ax2]) {
    ax1 = ax2;
    ax2 = 3 - q2 - ax1;
  }
  if (rottabs[3 * q2 + pax1] < rottabs[3 * q2 + pax2]) {
    pax1 = pax2;
    pax2 = 3 - q1 - pax1;
  }

  if (cle1 & (1 << pax2))
    clface = pax2;
  else
    clface = pax2 + 3;


  // printf("%lf - %d %d %d %d   %d %d     %d %d %d %d %d\n",
  //        penetration,cle1,cle2,code,in,q1,q2,clface,ax1,ax2,pax1,pax2);


  old_zero(rotmore, 9);
  if (clface == 0)
    rotmore[2] = -1, rotmore[4] = +1, rotmore[6] = +1;
  else if (clface == 1)
    rotmore[0] = +1, rotmore[5] = -1, rotmore[7] = +1;
  else if (clface == 2)
    rotmore[0] = +1, rotmore[4] = +1, rotmore[8] = +1;
  else if (clface == 3)
    rotmore[2] = +1, rotmore[4] = +1, rotmore[6] = -1;
  else if (clface == 4)
    rotmore[0] = +1, rotmore[5] = +1, rotmore[7] = -1;
  else if (clface == 5)
    rotmore[0] = -1, rotmore[4] = +1, rotmore[8] = -1;


  i0 = 0;
  i1 = 1;
  i2 = 2;
  f0 = f1 = f2 = 1;

  if (clface == 0) {
    i0 = 2;
    f0 = -1;
    i2 = 0;
  } else if (clface == 1) {
    i1 = 2;
    f1 = -1;
    i2 = 1;
  } else if (clface == 2) {
  } else if (clface == 3) {
    i0 = 2;
    i2 = 0;
    f2 = -1;
  } else if (clface == 4) {
    i1 = 2;
    i2 = 1;
    f2 = -1;
  } else if (clface == 5) {
    f0 = -1;
    f2 = -1;
  }

  // old_rotVecMat(p,pos21,rotmore);
  // old_rotVecMat(rnorm,clnorm,rotmore);
  rotaxis(p, pos21);
  rotaxis(rnorm, clnorm);

  // print("rnorm",rnorm);

  // old_mulMatMat(r,rotmore,rot,3,3,3);
  rotmatx(r, rot);

  old_rotVecMatT(tmp1, size1, rotmore);
  for (i = 0; i < 3; i++)
    s[i] = old_abs(tmp1[i]);

  old_transpose(rt, r, 3, 3);


  lx = s[0];
  ly = s[1];
  hz = s[2];
  p[2] -= hz;


  n = 0;
  old_copy3(points[n], p);
  old_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? 1 : -1));
  old_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
  old_copy3(points[n + 1], points[n]);
  old_addToScl3(points[n], rt + 3 * q2, size2[q2]);
  n = 1;
  old_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
  n = 2;


  old_copy3(points[n], p);
  old_addToScl3(points[n], rt + 3 * ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? -1 : 1));
  old_addToScl3(points[n], rt + 3 * ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
  old_copy3(points[n + 1], points[n]);
  old_addToScl3(points[n], rt + 3 * q2, size2[q2]);
  n = 3;
  old_addToScl3(points[n], rt + 3 * q2, -size2[q2]);
  n = 4;


  old_copy3(axi[0], points[0]);
  old_sub3(axi[1], points[1], points[0]);
  old_sub3(axi[2], points[2], points[0]);


  if (fabs(rnorm[2]) < mjMINVAL)
    return 0;  // shouldn't happen

  innorm = (1 / rnorm[2]) * (in ? -1 : 1);
  // printf("%lf\n",innorm);

  for (i = 0; i < 4; i++)
  {
    c1 = -points[i][2] * (1 / rnorm[2]);

    old_copy3(pu[i], points[i]);

    old_addToScl3(points[i], rnorm, c1);

    // ppts[i][0]=points[i][0];
    // ppts[i][1]=points[i][1];
    ppts2[i][0] = points[i][0];
    ppts2[i][1] = points[i][1];
  }


  old_copy3(pts[0], points[0]);
  old_sub3(pts[1], points[1], points[0]);
  old_sub3(pts[2], points[2], points[0]);

  m = 3;
  k = 0;
  n = 0;


  if (m > 1) {
    old_copy3(lines[k] + 0, pts[0]);
    old_copy3(lines[k] + 3, pts[1]);
    old_copy3(linesu[k] + 0, axi[0]);
    old_copy3(linesu[k++] + 3, axi[1]);
  }
  if (m > 2) {
    old_copy3(lines[k] + 0, pts[0]);
    old_copy3(lines[k] + 3, pts[2]);
    old_copy3(linesu[k] + 0, axi[0]);
    old_copy3(linesu[k++] + 3, axi[2]);

    old_add3(lines[k] + 0, pts[0], pts[1]);
    old_copy3(lines[k] + 3, pts[2]);
    old_add3(linesu[k] + 0, axi[0], axi[1]);
    old_copy3(linesu[k++] + 3, axi[2]);

    old_add3(lines[k] + 0, pts[0], pts[2]);
    old_copy3(lines[k] + 3, pts[1]);
    old_add3(linesu[k] + 0, axi[0], axi[2]);
    old_copy3(linesu[k++] + 3, axi[1]);
  }

  for (i = 0; i < k; i++) {
    for (q = 0; q < 2; q++) {
      a = lines[i][0 + q];
      b = lines[i][3 + q];
      c = lines[i][1 - q];
      d = lines[i][4 - q];

      if (fabs(b) > mjMINVAL) {
        for (j = -1; j <= 1; j += 2) {
          if (n < mjMAXCONPAIR) {
            l = s[q] * j;
            c1 = (l - a) * (1 / b);
            if (c1 < 0 || c1 > 1)
              continue;
            c2 = c + d * c1;
            if (fabs(c2) > s[1 - q])
              continue;

            if ((linesu[i][2] + linesu[i][5]*c1)*innorm > margin)
              continue;

            old_scl3(points[n], linesu[i], 0.5);
            old_addToScl3(points[n], linesu[i] + 3, 0.5 * c1);
            points[n][0 + q] += 0.5 * l;
            points[n][1 - q] += 0.5 * c2;
            depth[n] = points[n][2] * innorm * 2;
            n++;
          }
        }
      }
    }
  }

  nl = n;

  a = pts[1][0];
  b = pts[2][0];
  c = pts[1][1];
  d = pts[2][1];
  c1 = a * d - b * c;

  for (i = 0; i < 4; i++) {
    if (n < mjMAXCONPAIR) {
      llx = i / 2 ? lx : -lx;
      lly = i % 2 ? ly : -ly;

      x = llx - pts[0][0];
      y = lly - pts[0][1];

      u = (x * d - y * b) * (1 / c1);
      v = (y * a - x * c) * (1 / c1);

      if (nl == 0) {
        if ((u < 0 || u > 1) && (v < 0 || v > 1))
          continue;
      } else {
        if ((u < 0 || u > 1 || v < 0 || v > 1))
          continue;
      }

      if (u < 0)
        u = 0;
      if (u > 1)
        u = 1;
      if (v < 0)
        v = 0;
      if (v > 1)
        v = 1;


      old_scl3(tmp1, pu[0], 1 - u - v);
      old_addToScl3(tmp1, pu[1], u);
      old_addToScl3(tmp1, pu[2], v);

      points[n][0] = llx;
      points[n][1] = lly;
      points[n][2] = 0;

      old_sub3(tmp2, points[n], tmp1);

      c1 = old_dot3(tmp2, tmp2);
      if (tmp1[2] > 0)
        if (c1 > margin2)
          continue;

      old_add3(points[n], points[n], tmp1);
      old_scl3(points[n], points[n], 0.5);

      depth[n] = sqrt(c1) * (tmp1[2] < 0 ? -1 : 1);
      n++;
    }
  }

  nf = n;

  for (i = 0; i < 4; i++) {
    if (n < mjMAXCONPAIR) {
      x = ppts2[i][0];
      y = ppts2[i][1];

      if (nl == 0) {
        if (nf == 0) {
        } else {
          if (x < -lx || x > lx)
            if (y < -ly || y > ly)
              continue;
        }
      } else {
        if (x < -lx || x > lx || y < -ly || y > ly)
          continue;
      }

      for (c1 = 0, j = 0; j < 2; j++)
        if (ppts2[i][j] < -s[j])
          c1 += (ppts2[i][j] + s[j]) * (ppts2[i][j] + s[j]);
        else if (ppts2[i][j] > s[j])
          c1 += (ppts2[i][j] - s[j]) * (ppts2[i][j] - s[j]);

      c1 += pu[i][2] * innorm * pu[i][2] * innorm;

      if (pu[i][2] > 0)
        if (c1 > margin2)
          continue;


      tmp1[0] = ppts2[i][0] * 0.5;
      tmp1[1] = ppts2[i][1] * 0.5;
      tmp1[2] = 0;

      for (j = 0; j < 2; j++) {
        if (ppts2[i][j] < -s[j])
          tmp1[j] = -s[j] * 0.5;
        else if (ppts2[i][j] > s[j])
          tmp1[j] = +s[j] * 0.5;
      }
      old_addToScl3(tmp1, pu[i], 0.5);
      old_copy3(points[n], tmp1);

      depth[n] = sqrt(c1) * (pu[i][2] < 0 ? -1 : 1);
      n++;
    }
  }

  old_mulMatMatT(r, mat1, rotmore, 3, 3, 3);

  old_rotVecMat(tmp1, rnorm, r);

  old_scl3(con[0].frame, tmp1, in ? -1 : 1);
  old_zero3(con[0].frame + 3);


  for (i = 0; i < n; i++) {
    con[i].dist = depth[i];
    points[i][2] += hz;

    old_rotVecMat(tmp2, points[i], r);

    old_add3(con[i].pos, tmp2, pos1);

    old_copy(con[i].frame, con[0].frame, 6);
  }

  return n;

#undef rotaxis
#undef rotmatx
}

// ---- wrapper: 2.3.7 driver post-processing + mjPreContact output ----

// 0: mujoco 2.3.7 driver (exact-duplicate removal only)
// 1: mujoco 3.1.x-3.3.x driver (also drop contacts outside one box and not inside the other, ratio 1.01)
int compat_boxbox_badfilter = 0;
// max contacts the 3.14 driver reserves for a box-box pair (mj_maxContact returns 8)
int compat_boxbox_maxcon = 8;
// statistics
atomic_long compat_boxbox_ncall = 0;
atomic_long compat_boxbox_noverflow = 0;   // calls where >maxcon contacts remained after dedup
atomic_long compat_boxbox_nmax = 0;        // max contacts seen after dedup

static int old_outsideBox(const mjtNum point[3], const mjtNum pos[3], const mjtNum mat[9],
                          const mjtNum size[3], mjtNum inflate) {
  // mujoco 3.2.3 mju_outsideBox (inflate > 1 always here)
  mjtNum v0[3] = {point[0]-pos[0], point[1]-pos[1], point[2]-pos[2]};
  mjtNum vec[3] = {
    mat[0]*v0[0] + mat[3]*v0[1] + mat[6]*v0[2],
    mat[1]*v0[0] + mat[4]*v0[1] + mat[7]*v0[2],
    mat[2]*v0[0] + mat[5]*v0[1] + mat[8]*v0[2]};
  mjtNum big[3] = {size[0]*inflate, size[1]*inflate, size[2]*inflate};
  if (vec[0] > big[0] || vec[0] < -big[0] || vec[1] > big[1] || vec[1] < -big[1] ||
      vec[2] > big[2] || vec[2] < -big[2]) return 1;
  mjtNum small[3] = {size[0]/inflate, size[1]/inflate, size[2]/inflate};
  if (vec[0] < small[0] && vec[0] > -small[0] && vec[1] < small[1] && vec[1] > -small[1] &&
      vec[2] < small[2] && vec[2] > -small[2]) return -1;
  return 0;
}

int compat_BoxBox_old(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  oldcon_t buf[mjMAXCONPAIR];
  atomic_fetch_add(&compat_boxbox_ncall, 1);
  int num = old_BoxBox(m, d, buf, g1, g2, margin);
  if (num <= 0) return 0;

  for (int i = 0; i < num; i++) buf[i].bad = 0;

  if (compat_boxbox_badfilter) {
    const mjtNum* pos1 = d->geom_xpos + 3*g1; const mjtNum* mat1 = d->geom_xmat + 9*g1;
    const mjtNum* size1 = m->geom_size + 3*g1;
    const mjtNum* pos2 = d->geom_xpos + 3*g2; const mjtNum* mat2 = d->geom_xmat + 9*g2;
    const mjtNum* size2 = m->geom_size + 3*g2;
    for (int i = 0; i < num; i++) {
      mjtNum sz1[3] = {size1[0] + margin, size1[1] + margin, size1[2] + margin};
      mjtNum sz2[3] = {size2[0] + margin, size2[1] + margin, size2[2] + margin};
      int out1 = old_outsideBox(buf[i].pos, pos1, mat1, sz1, 1.01);
      int out2 = old_outsideBox(buf[i].pos, pos2, mat2, sz2, 1.01);
      if ((out1 == 1 && out2 != -1) || (out2 == 1 && out1 != -1)) buf[i].bad = 1;
    }
  }

  // duplicates: identical pos -> drop the earlier one (2.3.7 / 3.2.3 driver)
  for (int i = 0; i < num-1; i++) {
    if (buf[i].bad) continue;
    for (int j = i+1; j < num; j++) {
      if (buf[j].bad) continue;
      if (buf[i].pos[0] == buf[j].pos[0] && buf[i].pos[1] == buf[j].pos[1] &&
          buf[i].pos[2] == buf[j].pos[2]) {
        buf[i].bad = 1;
        break;
      }
    }
  }
  int n = 0;
  for (int j = 0; j < num; j++) {
    if (!buf[j].bad) {
      if (n < j) buf[n] = buf[j];
      n++;
    }
  }

  long prev = atomic_load(&compat_boxbox_nmax);
  while (n > prev && !atomic_compare_exchange_weak(&compat_boxbox_nmax, &prev, n)) {}

  // the 3.14 driver reserves only mj_maxContact()=8 precontacts for a box-box pair;
  // writing more would overrun the next pair's buffer. Keep maxcon of them:
  // deepest first, then greedy furthest-point (same rule as 3.14 filterPreContacts), original order kept.
  int maxcon = compat_boxbox_maxcon;
  if (n > maxcon) {
    atomic_fetch_add(&compat_boxbox_noverflow, 1);
    int sel[mjMAXCONPAIR] = {0};
    mjtNum mind[mjMAXCONPAIR];
    int best = 0;
    for (int i = 0; i < n; i++) { mind[i] = 1e300; if (buf[i].dist < buf[best].dist) best = i; }
    for (int k = 0; k < maxcon && best >= 0; k++) {
      sel[best] = 1;
      int next = -1; mjtNum nextd = -1;
      for (int i = 0; i < n; i++) {
        if (sel[i]) continue;
        mjtNum dx = buf[i].pos[0]-buf[best].pos[0], dy = buf[i].pos[1]-buf[best].pos[1],
               dz = buf[i].pos[2]-buf[best].pos[2];
        mjtNum d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < mind[i]) mind[i] = d2;
        if (mind[i] > nextd) { nextd = mind[i]; next = i; }
      }
      best = next;
    }
    int k = 0;
    for (int i = 0; i < n; i++) if (sel[i]) { if (k < i) buf[k] = buf[i]; k++; }
    n = k;
  }

  for (int i = 0; i < n; i++) {
    con[i].dist = buf[i].dist;
    con[i].pos[0] = buf[i].pos[0]; con[i].pos[1] = buf[i].pos[1]; con[i].pos[2] = buf[i].pos[2];
    // 2.3.7 sets frame[0:3] = normal (unit, from a rotation matrix column / normalized clnorm),
    // frame[3:6] = 0; mju_makeFrame later normalizes and picks the tangent; 3.14 copies
    // normal->frame[0:3], tangent->frame[3:6] and calls the same mju_makeFrame logic.
    con[i].normal[0] = buf[i].frame[0]; con[i].normal[1] = buf[i].frame[1]; con[i].normal[2] = buf[i].frame[2];
    con[i].tangent[0] = buf[i].frame[3]; con[i].tangent[1] = buf[i].frame[4]; con[i].tangent[2] = buf[i].frame[5];
  }
  return n;
}


// ---- mujoco 2.3.7 quaternion normalization (mj_kinematics prologue) ----

atomic_long compat_normquat_ncall = 0;

static mjtNum old_normalize4(mjtNum vec[4]) {
  mjtNum norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2] + vec[3]*vec[3]);
  if (norm < mjMINVAL) {
    vec[0] = 1;
    vec[1] = 0;
    vec[2] = 0;
    vec[3] = 0;
  } else {
    mjtNum normInv = 1/norm;
    vec[0] *= normInv;
    vec[1] *= normInv;
    vec[2] *= normInv;
    vec[3] *= normInv;
  }
  return norm;
}

void compat_normalizeQuat_237(const mjModel* m, mjData* d) {
  atomic_fetch_add(&compat_normquat_ncall, 1);
  for (int i=0; i < m->njnt; i++) {
    if (m->jnt_type[i] == mjJNT_BALL || m->jnt_type[i] == mjJNT_FREE) {
      old_normalize4(d->qpos+m->jnt_qposadr[i]+3*(m->jnt_type[i] == mjJNT_FREE));
    }
  }
  for (int i=0; i < m->nmocap; i++) {
    old_normalize4(d->mocap_quat+4*i);
  }
}
