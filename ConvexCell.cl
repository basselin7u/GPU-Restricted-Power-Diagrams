#if (VHSAlgo==2) && (__OPENCL_VERSION__ < 120)
#  if cl_khr_fp64
#    pragma OPENCL EXTENSION cl_khr_fp64: enable
#  else
#    error Missing double precision extension
#  endif
#endif

#include <cl_kernel.h>

#if defined(CHESSIAN) || (defined(CUSED_CELL) && (CUSED_CELL_DIST>0))
#  define NEED_PLANE_NEIGHBOURS
#endif

#ifdef DEBUG_COUNTER0
#  define DCOUNTER0(x) x
#else
#  define DCOUNTER0(x) 
#endif
#ifdef DEBUG_COUNTER1
#  define DCOUNTER1(x) x
#else
#  define DCOUNTER1(x) 
#endif
#ifdef DEBUG_COUNTER2
#  define DCOUNTER2(x) x
#else
#  define DCOUNTER2(x) 
#endif
#ifdef DEBUG_COUNTER3
#  define DCOUNTER3(x) x
#else
#  define DCOUNTER3(x) 
#endif
#ifdef DEBUG_COUNTER4
#  define DCOUNTER4(x) x
#else
#  define DCOUNTER4(x) 
#endif

// basic.h
#define swapUChar3(__a,__b) { uchar3 __t = __a; __a = __b; __b = __t; }
#define swapReal4(__a,__b) { real4 __t = __a; __a = __b; __b = __t; }

#  define make_real4 (real4)
#ifndef make_uchar3
#  define make_uchar3 (uchar3)
#endif

#define PL (P-P1)
real dot3(real4 A, real4 B);
real4 mul3(real s, real4 A);
real det2x2(real a11, real a12, real a21, real a22);
real det3x3(real a11, real a12, real a13, real a21, real a22, real a23, real a31, real a32, real a33);
real det3x3R4(real4 A, real4 B, real4 C);
real det4x4(real a11, real a12, real a13, real a14,
	     real a21, real a22, real a23, real a24,               
	     real a31, real a32, real a33, real a34,  
	     real a41, real a42, real a43, real a44);
real4 get_plane_from_points(real4 A, real4 B, real4 C);

real dot3(real4 A, real4 B) {
    return A.x*B.x + A.y*B.y + A.z*B.z;
}
real4 mul3(real s, real4 A) {
    return make_real4(s*A.x, s*A.y, s*A.z, 1.);
}

real det2x2(real a11, real a12, real a21, real a22) {
    return a11*a22 - a12*a21;
}
real det3x3(real a11, real a12, real a13, real a21, real a22, real a23, real a31, real a32, real a33) {
    return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}
real det3x3R4(real4 A, real4 B, real4 C) {
    return det3x3(A.x,A.y,A.z,B.x,B.y,B.z,C.x,C.y,C.z);
}

real det4x4(real a11, real a12, real a13, real a14,
	     real a21, real a22, real a23, real a24,               
	     real a31, real a32, real a33, real a34,  
	     real a41, real a42, real a43, real a44)
{
    real const m12 = a21*a12 - a11*a22;
    real const m13 = a31*a12 - a11*a32;
    real const m14 = a41*a12 - a11*a42;
    real const m23 = a31*a22 - a21*a32;
    real const m24 = a41*a22 - a21*a42;
    real const m34 = a41*a32 - a31*a42;
    
    real const m123 = m23*a13 - m13*a23 + m12*a33;
    real const m124 = m24*a13 - m14*a23 + m12*a43;
    real const m134 = m34*a13 - m14*a33 + m13*a43;
    real const m234 = m34*a23 - m24*a33 + m23*a43;
    
    return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}

#if VHSAlgo==2
double det4x4D(double a11, double a12, double a13, double a14,
	       double a21, double a22, double a23, double a24,               
	       double a31, double a32, double a33, double a34,  
	       double a41, double a42, double a43, double a44);
double det4x4D(double a11, double a12, double a13, double a14,
	       double a21, double a22, double a23, double a24,               
	       double a31, double a32, double a33, double a34,  
	       double a41, double a42, double a43, double a44)
{
    double const m12 = a21*a12 - a11*a22;
    double const m13 = a31*a12 - a11*a32;
    double const m14 = a41*a12 - a11*a42;
    double const m23 = a31*a22 - a21*a32;
    double const m24 = a41*a22 - a21*a42;
    double const m34 = a41*a32 - a31*a42;
    
    double const m123 = m23*a13 - m13*a23 + m12*a33;
    double const m124 = m24*a13 - m14*a23 + m12*a43;
    double const m134 = m34*a13 - m14*a33 + m13*a43;
    double const m234 = m34*a23 - m24*a33 + m23*a43;
    
    return (m234*a14 - m134*a24 + m124*a34 - m123*a44);
}
#endif

real4 get_plane_from_points(real4 A, real4 B, real4 C) {
    real4 plane = cross(B-A, C-A);
    plane.w = -dot3(plane, A);
    return plane;
}

// ConvexCell
enum { END_OF_LIST = 255 };

typedef struct {
    real4 voro_seed;
    __global Status *status;
    __global real4 const * pts;
    __local uchar4 * m_vertices;
    __local uchar *m_next_boundaries;
    __local real * m_planes;
#if P1>0
    __global real * m_planes2;
#endif
#ifdef NEED_PLANE_NEIGHBOURS
    __local int * m_planesNeighbour;
#endif
    uchar nb_vertex;
    uchar nb_plane;
} ConvexCell;

// accessor

__local uchar *CCBoundaryNext(ConvexCell *cc, int p);
real4 CCPlane(ConvexCell *cc, int p);
void CCSetPlaneR4(ConvexCell *cc, int p, real4 plane);
void CCSetPlane(ConvexCell *cc, int p, real x, real y, real z, real w);
__local uchar3 *CCVertex(ConvexCell *cc, int v);
#ifdef NEED_PLANE_NEIGHBOURS
__local int *CCPlaneNeighbour(ConvexCell *cc, int p);
#endif

// ConvexCell functions
void CCInit(ConvexCell *cc, int id, __global real4 const * p_pts, __global Status* p_status);
void CCClipByPlane(ConvexCell *cc, int vid);
void CCClipTetFromPointsInfinite(ConvexCell *cc, real4 A, real4 B, real4 C, real4 D);
uchar CCComputeBoundary(ConvexCell *cc, uchar nb_removed);
real4 CCComputeVertexCoordinates(ConvexCell *cc, uchar3 t, bool persp_divide);
void CCExportBaryVolume(ConvexCell *cc, __global real4 * vol_bary, real factor);
bool CCIsSecurityRadiusReached(ConvexCell *cc, real4 last_neig);
uchar CCIthPlane(ConvexCell *cc, uchar t, int i);
int CCNewPlane(ConvexCell *cc, int vid);
void CCNewVertex(ConvexCell *cc, uchar i, uchar j, uchar k);
void CCUpdateUsedCell(ConvexCell *cc, __global uchar *usedCells, int d);
bool CCVertexIsInHalfSpace(ConvexCell *cc, uchar3 t, real4 eqn);
#if P1>0
void CCRenumberPlanes(ConvexCell *cc);
#endif

__local uchar *CCBoundaryNext(ConvexCell *cc, int p)
{
    return &cc->m_next_boundaries[((p&0x7ffffffc)*get_local_size(0))|(p&0x3)];
}

real4 CCPlane(ConvexCell *cc, int p)
{
#if P1>0
  if (p<PL) {
    int localSize=get_local_size(0);
    __local real *ptr=&cc->m_planes[4*p*localSize];
    return make_real4(ptr[0],ptr[localSize],ptr[2*localSize],ptr[3*localSize]);
  }
  p-=PL;
  return make_real4(cc->m_planes2[4*p],cc->m_planes2[4*p+1],cc->m_planes2[4*p+2],cc->m_planes2[4*p+3]);
#else
    int localSize=get_local_size(0);
    __local real *ptr=&cc->m_planes[4*p*localSize];
    return make_real4(ptr[0],ptr[localSize],ptr[2*localSize],ptr[3*localSize]);
#endif
}

void CCSetPlane(ConvexCell *cc, int p, real x, real y, real z, real w) {
#if P1>0
    if (p<PL) {
        int localSize=get_local_size(0);
        __local real *ptr=&cc->m_planes[4*p*localSize];
        ptr[0]=x;
        ptr[localSize]=y;
        ptr[2*localSize]=z;
        ptr[3*localSize]=w;
        return;
    }
    p-=PL;
    __global real *ptr=&cc->m_planes2[4*p];
    ptr[0]=x;
    ptr[1]=y;
    ptr[2]=z;
    ptr[3]=w;
#else
    int localSize=get_local_size(0);
    __local real *ptr=&cc->m_planes[4*p*localSize];
    ptr[0]=x;
    ptr[localSize]=y;
    ptr[2*localSize]=z;
    ptr[3*localSize]=w;
#endif
}

void CCSetPlaneR4(ConvexCell *cc, int p, real4 plane) {
    CCSetPlane(cc, p, plane.x, plane.y, plane.z, plane.w);
}

__local uchar3 *CCVertex(ConvexCell *cc, int v)
{
    return (__local uchar3 *) &(cc->m_vertices[v*get_local_size(0)]);
}

#if P1>0
void CCRenumberPlanes(ConvexCell *cc)
{
    for (int i=0; i<cc->nb_plane; ++i)
      *CCBoundaryNext(cc,i)=255;
    for (int i=0; i<cc->nb_vertex; ++i) {
      uchar3 pl=*CCVertex(cc, i);
      *CCBoundaryNext(cc,pl.x)=0;
      *CCBoundaryNext(cc,pl.y)=0;
      *CCBoundaryNext(cc,pl.z)=0;
    }
    int n=0;
    for (int i=0; i<cc->nb_plane; ++i) {
      __local uchar *c=CCBoundaryNext(cc,i);
      if (*c!=0) continue;
      if (i!=n) {
	CCSetPlaneR4(cc, n, CCPlane(cc, i));	
#ifdef NEED_PLANE_NEIGHBOURS
	*CCPlaneNeighbour(cc,n)=*CCPlaneNeighbour(cc,i);
#endif
      }
      *c=n++;
    }
    for (int i=0; i<cc->nb_vertex; ++i) {
      uchar3 pl=*CCVertex(cc, i);
      pl.x=*CCBoundaryNext(cc,pl.x);
      pl.y=*CCBoundaryNext(cc,pl.y);
      pl.z=*CCBoundaryNext(cc,pl.z);
      *CCVertex(cc, i)=pl;
    }
    cc->nb_plane=n;
}
#endif

#ifdef NEED_PLANE_NEIGHBOURS
__local int *CCPlaneNeighbour(ConvexCell *cc, int p)
{
    return &(cc->m_planesNeighbour[p*get_local_size(0)]);
}
#endif

void CCInit(ConvexCell *cc, int id, __global real4 const * p_pts, __global Status* p_status)
{
    cc->status = p_status;
    cc->pts = p_pts;
    cc->voro_seed=p_pts[id];
    
#ifdef NEED_PLANE_NEIGHBOURS
    for(int i=0; i<P; ++i)
        *CCPlaneNeighbour(cc, i)=-1;
#endif
    real const eps  = CUBE_EPSILON;
    real const vmin = -eps;
    real const vmax = 1000 + eps;
    CCSetPlane(cc, 0, 1.0,  0.0,  0.0, -vmin);
    CCSetPlane(cc, 1,-1.0,  0.0,  0.0,  vmax);
    CCSetPlane(cc, 2, 0.0,  1.0,  0.0, -vmin);
    CCSetPlane(cc, 3, 0.0, -1.0,  0.0,  vmax);
    CCSetPlane(cc, 4, 0.0,  0.0,  1.0, -vmin);
    CCSetPlane(cc, 5, 0.0,  0.0, -1.0,  vmax);
    cc->nb_plane = 6;

    *CCVertex(cc, 0) = make_uchar3(2, 5, 0);
    *CCVertex(cc, 1) = make_uchar3(5, 3, 0);
    *CCVertex(cc, 2) = make_uchar3(1, 5, 2);
    *CCVertex(cc, 3) = make_uchar3(5, 1, 3);
    *CCVertex(cc, 4) = make_uchar3(4, 2, 0);
    *CCVertex(cc, 5) = make_uchar3(4, 0, 3);
    *CCVertex(cc, 6) = make_uchar3(2, 4, 1);
    *CCVertex(cc, 7) = make_uchar3(4, 3, 1);
    cc->nb_vertex = 8;
}

uchar CCIthPlane(ConvexCell *cc, uchar t, int i)
{
    return ((__local uchar *)&(*CCVertex(cc, t)))[i];
}
int CCNewPlane(ConvexCell *cc, int vid)
{
    if (cc->nb_plane >= P) { 
        *cc->status = plane_overflow; 
        return -1; 
    }
    
    real4 B = cc->pts[vid];
    real4 dir = cc->voro_seed-B;
    real4 ave2 = cc->voro_seed+B;
    real dot = dot3(ave2,dir) - (B.w - cc->voro_seed.w); // we add B.w-voro_seed.w for power diagram
    if (B.w<=NO_CELL_LIMITS) dot = 2*dot3(B,dir); // needed as outside points have B.w=NO_CELL_LIMITS
    // we need to normalize the normal to compute the intrinsic
    real dirNorm=sqrt(dot3(dir,dir));
    CCSetPlane(cc,cc->nb_plane,dir.x/dirNorm, dir.y/dirNorm, dir.z/dirNorm, -dot / 2 /dirNorm);
#ifdef NEED_PLANE_NEIGHBOURS
    *CCPlaneNeighbour(cc, cc->nb_plane)=vid;
#endif
    return cc->nb_plane++;
}

void CCNewVertex(ConvexCell *cc, uchar i, uchar j, uchar k)
{
    if (cc->nb_vertex >= V) {
        *cc->status = vertex_overflow; 
        return; 
    }
    *CCVertex(cc, cc->nb_vertex++) = make_uchar3(i, j, k);
}

real4 CCComputeVertexCoordinates(ConvexCell *cc, uchar3 t, bool persp_divide)
{
    real4 const pi1 = CCPlane(cc, t.x);
    real4 const pi2 = CCPlane(cc, t.y);
    real4 const pi3 = CCPlane(cc, t.z);
    real4 result;
    real const m12=pi2.x*pi3.y-pi2.y*pi3.x;
    real const m13=pi2.x*pi3.z-pi2.z*pi3.x;
    real const m14=pi2.x*pi3.w-pi2.w*pi3.x;
    real const m23=pi2.y*pi3.z-pi2.z*pi3.y;
    real const m24=pi2.y*pi3.w-pi2.w*pi3.y;
    real const m34=pi2.z*pi3.w-pi2.w*pi3.z;
    result.x = -pi1.w*m23-pi1.y*m34+pi1.z*m24;
    result.y = pi1.x*m34+pi1.w*m13-pi1.z*m14;
    result.z = -pi1.x*m24+pi1.y*m14-pi1.w*m12;
    result.w = pi1.x*m23-pi1.y*m13+pi1.z*m12;
    if (persp_divide) return make_real4(result.x / result.w, result.y / result.w, result.z / result.w, 1);
    return result;
}

bool CCIsSecurityRadiusReached(ConvexCell *cc, real4 last_neig)
{
    // finds furthest voro vertex distance2
    real v_dist = 0;
    for(int i=0; i<cc->nb_vertex; ++i) {
        real4 pc = CCComputeVertexCoordinates(cc, *CCVertex(cc, i), true);
        real4 diff = pc-cc->voro_seed;
        real d2 = dot3(diff, diff); // TODO safe to put dot4 here, diff.w = 0
        v_dist = max(d2, v_dist);
    }
    //compare to new neighbors distance2
    real4 diff = last_neig-cc->voro_seed; // TODO it really should take index of the neighbor instead of the real4, then would be safe to put dot
    real d2 = dot3(diff, diff);
    //
    // change the following line if you want to compute power diagram to
    // return (d2 > 6*v_dist);
    //
    return (d2 > 4*v_dist);
}

#if VHSAlgo==1
real fmax3(real a, real b, real c);
real fmax4(real a, real b, real c, real d);
real fmaxAbs(real a, real b);
real fmax3(real a, real b, real c) {
    return fmax(fmax(a,b),c);
}
real fmax4(real a, real b, real c, real d) {
    return fmax(fmax(a,b),fmax(c,d));
}
real fmaxAbs(real a, real b)
{
    return fmax(fabs(a),fabs(b));
}
#endif

bool CCVertexIsInHalfSpace(ConvexCell *cc, uchar3 t, real4 eqn)
{
#if VHSAlgo==2
    real4 pi1 = CCPlane(cc, t.x);
    real4 pi2 = CCPlane(cc, t.y);
    real4 pi3 = CCPlane(cc, t.z);

    double det = det4x4D(
                        (double)pi1.x, (double)pi2.x, (double)pi3.x, (double)eqn.x,
                        (double)pi1.y, (double)pi2.y, (double)pi3.y, (double)eqn.y,
                        (double)pi1.z, (double)pi2.z, (double)pi3.z, (double)eqn.z,
                        (double)pi1.w, (double)pi2.w, (double)pi3.w, (double)eqn.w
                        );

    return (det > 0.0);
#elif VHSAlgo==1
    real4 planes[3]={CCPlane(cc, t.x),CCPlane(cc, t.y),CCPlane(cc, t.z)};
    real tmp;
    real m12 = planes[1].x*planes[0].y; tmp=-planes[0].x*planes[1].y;
    real M12 = fmaxAbs(m12,tmp); m12 += tmp;
    real m13 = planes[2].x*planes[0].y; tmp=-planes[0].x*planes[2].y;
    real M13 = fmaxAbs(m13,tmp); m13 += tmp;
    real m14 = eqn.x*planes[0].y; tmp=-planes[0].x*eqn.y;
    real M14 = fmaxAbs(m14,tmp); m14 += tmp;
    real m23 = planes[2].x*planes[1].y; tmp=-planes[1].x*planes[2].y;
    real M23 = fmaxAbs(m23,tmp); m23 += tmp;
    real m24 = eqn.x*planes[1].y; tmp=-planes[1].x*eqn.y;
    real M24 = fmaxAbs(m24,tmp); m24 += tmp;
    real m34 = eqn.x*planes[2].y; tmp=-planes[2].x*eqn.y;
    real M34 = fmaxAbs(m34,tmp); m34 += tmp;

    real m123 = m23*planes[0].z-m13*planes[1].z+m12*planes[2].z;
    real M123 = fmax3(fabs(M23*planes[0].z),fabs(M13*planes[1].z),fabs(M12*planes[2].z));
    real m124 = m24*planes[0].z-m14*planes[1].z+m12*eqn.z;
    real M124 = fmax3(fabs(M24*planes[0].z),fabs(M14*planes[1].z),fabs(M12*eqn.z));
    real m134 = m34*planes[0].z-m14*planes[2].z+m13*eqn.z;
    real M134 = fmax3(fabs(M34*planes[0].z),fabs(M14*planes[2].z),fabs(M13*eqn.z));
    real m234 = m34*planes[1].z-m24*planes[2].z+m23*eqn.z;
    real M234 = fmax3(fabs(M34*planes[1].z),fabs(M24*planes[2].z),fabs(M23*eqn.z));

    real res=m234*planes[0].w-m134*planes[1].w+m124*planes[2].w-m123*eqn.w;
    real M1234=fmax4(fabs(M234*planes[0].w),fabs(M134*planes[1].w),fabs(M124*planes[2].w),fabs(M123*eqn.w));
    if(fabs(res) < 3.e-6 * M1234) { // ~=23/2^23 (2.e-07=1/2^23 maybe ok)
        *cc->status = needs_exact_predicates;
        // if the point is very close to the plane, returns true, the vertex will be removed, ...
        return res > -2.e-7 * M1234;
    }

    return (res > 0.0f);
#else
    real4 pi1 = CCPlane(cc, t.x);
    real4 pi2 = CCPlane(cc, t.y);
    real4 pi3 = CCPlane(cc, t.z);
    real det = det4x4(
                       pi1.x, pi2.x, pi3.x, eqn.x,
                       pi1.y, pi2.y, pi3.y, eqn.y,
                       pi1.z, pi2.z, pi3.z, eqn.z,
                       pi1.w, pi2.w, pi3.w, eqn.w
                       );
    return (det > 0);
#endif
}

uchar CCComputeBoundary(ConvexCell *cc, uchar nb_removed)
{
    // clean circular list of the boundary
    for (int i=0; i<P; ++i) *CCBoundaryNext(cc, i) = END_OF_LIST;
    uchar firstBoundary = END_OF_LIST;

    int nb_iter = 0;
    uchar t = cc->nb_vertex;

    while (nb_removed>0) {
        if (nb_iter++>1000) { 
            *cc->status = inconsistent_boundary; 
            return firstBoundary; 
        }
        bool is_in_border[3];
        bool next_is_opp[3];
        for (int e=0; e<3; ++e) is_in_border[e] = (*CCBoundaryNext(cc, CCIthPlane(cc, t, e)) != END_OF_LIST);
        for (int e=0; e<3; ++e) next_is_opp[e] = (*CCBoundaryNext(cc, CCIthPlane(cc, t, (e + 1) % 3)) == CCIthPlane(cc, t, e));

        bool new_border_is_simple = true;
        // check for non manifoldness
        for (int e=0; e<3; ++e) if (!next_is_opp[e] && !next_is_opp[(e + 1) % 3] && is_in_border[(e + 1) % 3]) new_border_is_simple = false;
        // check for more than one boundary ... or first triangle
        if (!next_is_opp[0] && !next_is_opp[1] && !next_is_opp[2]) {
            if (firstBoundary == END_OF_LIST) {
                for (int e=0; e<3; ++e) *CCBoundaryNext(cc, CCIthPlane(cc, t, e)) = CCIthPlane(cc, t, (e + 1) % 3);
                firstBoundary = CCVertex(cc, t)->x;
            }
            else new_border_is_simple = false;
        }

        if (!new_border_is_simple) {
            t++;
            if (t == cc->nb_vertex + nb_removed) t = cc->nb_vertex;
            continue;
        }

        // link next
        for (int e=0; e<3; ++e) if (!next_is_opp[e]) *CCBoundaryNext(cc, CCIthPlane(cc, t, e)) = CCIthPlane(cc, t, (e + 1) % 3);

        // destroy link from removed vertices
        for (int e=0; e<3; ++e) if (next_is_opp[e] && next_is_opp[(e + 1) % 3]) {
            if (firstBoundary == CCIthPlane(cc, t, (e + 1) % 3)) firstBoundary = *CCBoundaryNext(cc, CCIthPlane(cc, t, (e + 1) % 3));
            *CCBoundaryNext(cc, CCIthPlane(cc, t, (e + 1) % 3)) = END_OF_LIST;
        }

        //remove triangle from R, and restart iterating on R
        swapUChar3(*CCVertex(cc, t), *CCVertex(cc, cc->nb_vertex+nb_removed-1));
        t = cc->nb_vertex;
        nb_removed--;
    }
    return firstBoundary;
}

void CCClipByPlane(ConvexCell *cc, int vid)
{
    if (*cc->status == plane_overflow) return;
    real4 eqn = CCPlane(cc, vid);
    uchar nb_removed = 0;

    int i = 0;
    while (i < cc->nb_vertex) { // for all vertices of the cell
        if(CCVertexIsInHalfSpace(cc, *CCVertex(cc, i), eqn)) {
            cc->nb_vertex--;
            swapUChar3(*CCVertex(cc, i), *CCVertex(cc, cc->nb_vertex));
            nb_removed++;
        }
        else i++;
    }
    //check if the cell is empty after cutting
    if (cc->nb_vertex < 1) {
        *cc->status = empty_cell;
        return;
    }

    if (*cc->status == needs_exact_predicates)
        return;

    // if no clips, then remove the plane equation
    if (nb_removed == 0) {
        --cc->nb_plane;
#ifdef NEED_PLANE_NEIGHBOURS
        *CCPlaneNeighbour(cc, cc->nb_plane)=-1;
#endif
        return;
    }

    // Step 2: compute cavity boundary
    uchar firstBoundary=CCComputeBoundary(cc,nb_removed);
    if (*cc->status != success) return;
    if (firstBoundary == END_OF_LIST) return;

    // Step 3: Triangulate cavity
    uchar cir = firstBoundary;
    do {
	    int newCir=*CCBoundaryNext(cc, cir);
	    CCNewVertex(cc, vid, cir, newCir);
        if (*cc->status != success) return;
        cir = newCir;
    } while (cir != firstBoundary);
}

void CCClipTetFromPointsInfinite(ConvexCell *cc, real4 A, real4 B, real4 C, real4 D)
{
    if (cc->nb_plane >= P) {*cc->status = plane_overflow; return ; }
    CCSetPlaneR4(cc, cc->nb_plane++, get_plane_from_points(A, B, C)); CCClipByPlane(cc, cc->nb_plane-1); if (*cc->status != success) return;
    if (cc->nb_plane >= P) {*cc->status = plane_overflow; return ; }
    CCSetPlaneR4(cc, cc->nb_plane++, get_plane_from_points(A, D, B)); CCClipByPlane(cc, cc->nb_plane-1); if (*cc->status != success) return;
    if (cc->nb_plane >= P) {*cc->status = plane_overflow; return ; }
    CCSetPlaneR4(cc, cc->nb_plane++, get_plane_from_points(A, C, D)); CCClipByPlane(cc, cc->nb_plane-1); if (*cc->status != success) return;
    if (cc->nb_plane >= P) {*cc->status = plane_overflow; return ; }
    CCSetPlaneR4(cc, cc->nb_plane++, get_plane_from_points(C, D, B)); CCClipByPlane(cc, cc->nb_plane-1); if (*cc->status != success) return;
}

void CCExportBaryVolume(ConvexCell *cc, __global real4 * vol_bary, real factor)
{
    real4 bary_sum=make_real4(0, 0, 0, 0);
    real4 planes[3];
    real4 APlaneI[3];
    real4 ALineII1[3];
    real4 const C = cc->voro_seed;
    uchar  planesId[3];
    for(int ve=0; ve<cc->nb_vertex; ++ve) {
        real4 const A = CCComputeVertexCoordinates(cc, *CCVertex(cc, ve), true);
        real4 const AC = C-A;
        
        for (int i=0; i<3; ++i) {
            planesId[i]=CCIthPlane(cc, ve, i);
            planes[i]=CCPlane(cc, planesId[i]);
        }
        for (int i=0; i<3; ++i) {
            real4 const ni=planes[i];
            APlaneI[i]=AC-dot3(AC,ni)/dot3(ni,ni)*ni;
        }
        for (int i=0; i<3; ++i) {
            real4 const ninj=cross(planes[i],planes[(i+1)%3]);
            real ninj2=dot(ninj,ninj);
            if (ninj2<VOLUME_EPSILON2*dot3(planes[i],planes[i])*dot3(planes[(i+1)%3],planes[(i+1)%3]))
                ALineII1[i]=make_real4(0, 0, 0, 0);
            else
                ALineII1[i]=dot(AC,ninj)/ninj2*ninj;
        }
        for (int i=0; i<3; ++i) {
            for (int st=0; st<2; ++st) {
                // A,C,APli,ALineII1j
                int const j=(i+3-st)%3;
                real const w=(2*st-1)*det3x3R4(AC,APlaneI[i],ALineII1[j]);
                bary_sum.w += w;
                real4 sum=AC+APlaneI[i]+ALineII1[j];
                bary_sum.x+=w*(A.x+0.25*sum.x);
                bary_sum.y+=w*(A.y+0.25*sum.y);
                bary_sum.z+=w*(A.z+0.25*sum.z);
            }
        }
    }
    *vol_bary = *vol_bary+factor/6*bary_sum;
}

#if defined(CHESSIAN)
static void CCExportBaryVolumeAndArea(ConvexCell *cc, __global real4 * vol_bary, int *neigId, real *surf_intr, real factor)
{
    real4 bary_sum=make_real4(0, 0, 0, 0);
    real4 planes[3];
    real4 APlaneI[3];
    real4 ALineII1[3];
    real4 const C = cc->voro_seed;
    uchar  planesId[3];
    for(int p=0; p<cc->nb_plane; ++p) neigId[p]=*CCPlaneNeighbour(cc, p);
    for(int ve=0; ve<cc->nb_vertex; ++ve) {
        real4 const A = CCComputeVertexCoordinates(cc, *CCVertex(cc, ve), true);
        real4 const AC = C-A;
        
        for (int i=0; i<3; ++i) {
            planesId[i]=CCIthPlane(cc, ve, i);
            planes[i]=CCPlane(cc, planesId[i]);
        }
        for (int i=0; i<3; ++i) {
            real4 const ni=planes[i];
            APlaneI[i]=AC-dot3(AC,ni)/dot3(ni,ni)*ni;
        }
        for (int i=0; i<3; ++i) {
            real4 const ninj=cross(planes[i],planes[(i+1)%3]);
            real ninj2=dot(ninj,ninj);
            if (ninj2<VOLUME_EPSILON2*dot3(planes[i],planes[i])*dot3(planes[(i+1)%3],planes[(i+1)%3]))
                ALineII1[i]=make_real4(0, 0, 0, 0);
            else
                ALineII1[i]=dot(AC,ninj)/ninj2*ninj;
        }
        for (int i=0; i<3; ++i) {
	    for (int st=0; st<2; ++st) {
                // A,C,APli,ALineII1j
                int const j=(i+3-st)%3;
                real const w=(2*st-1)*det3x3R4(AC,APlaneI[i],ALineII1[j]);
                bary_sum.w += w;
                real4 sum=AC+APlaneI[i]+ALineII1[j];
                bary_sum.x+=w*(A.x+0.25*sum.x);
                bary_sum.y+=w*(A.y+0.25*sum.y);
                bary_sum.z+=w*(A.z+0.25*sum.z);

                // check sign
                real triArea=(2*st-1)*factor*det3x3R4(APlaneI[i],ALineII1[j],planes[i])/2;
                sum=APlaneI[i]+ALineII1[j];
#if HC==1
                surf_intr[planesId[i]]+=triArea;
#else
                surf_intr[4*planesId[i]+0]+=triArea*(A.x+sum.x/3);
                surf_intr[4*planesId[i]+1]+=triArea*(A.y+sum.y/3);
                surf_intr[4*planesId[i]+2]+=triArea*(A.z+sum.z/3);
                surf_intr[4*planesId[i]+3]+=triArea;
#endif
            }
        }
    }
    *vol_bary = *vol_bary+factor/6*bary_sum;
}
#endif

void CCUpdateUsedCell(ConvexCell *cc, __global uchar *usedCells, int d)
{
#ifdef NEED_PLANE_NEIGHBOURS
    for(int ve=0; ve<cc->nb_vertex; ++ve) {
        for (int i=0; i<3; ++i) {
            int neigh=*CCPlaneNeighbour(cc, CCIthPlane(cc, ve, i));
            if (neigh>=0 && usedCells[neigh]==0) usedCells[neigh]=d;
        }
    }
#endif
}

typedef struct {
    Status status;
    uchar nb_vertex;
    uchar nb_plane;
    uchar3 m_vertices[V];
} ConvexCellBackUp;

void CCBUInit(ConvexCellBackUp *backup, ConvexCell *cc);
void CCBURestore(ConvexCellBackUp const *backup, ConvexCell *cc);

void CCBUInit(ConvexCellBackUp *backup, ConvexCell *cc)
{
    backup->nb_vertex = cc->nb_vertex;
    backup->nb_plane = cc->nb_plane;
    for(int i=0; i<cc->nb_vertex; ++i) backup->m_vertices[i] = *CCVertex(cc, i);
    backup->status = *cc->status;
}
void CCBURestore(ConvexCellBackUp const *backup, ConvexCell *cc)
{
    cc->nb_vertex = backup->nb_vertex;
    cc->nb_plane = backup->nb_plane;
    for(int i=0; i<cc->nb_vertex; ++i) *CCVertex(cc, i) = backup->m_vertices[i];
    *cc->status = backup->status;
}

#ifdef USE_GRID
int gridId(int x, int y, int z);
int gridId(int x, int y, int z) { return x + y*GRID_SIZE + z*GRID_SIZE *GRID_SIZE; }

#  if T>0
static bool merge(int const *list1, int n1, __global const int *list2, int n2, int *res, int *n)
{
    int remain[2];
    remain[0]=n1;
    remain[1]=n2;
    *n=0;
    for(int i=0; i<n1+n2; ++i) {
        if (remain[0] && remain[1] && list1[0]==list2[0]) {
            --remain[0];
            ++list1;
            ++i;
        }
        if (*n >= T) {
            printf(".");
            return false;
        }
        if (remain[0]!=0 && (remain[1]==0 || (list1[0]<list2[0]))) {
            *(res++)=*(list1++);
            --remain[0];
        }
        else {
            *(res++)=*(list2++);
            --remain[1];
        }
        ++*n;
    }
    return true;
}
#  endif

#endif

#if defined(CUSED_CELL) && (CUSED_CELL_DIST>1)
// small code, find bordering cells
__kernel void compute_bordering(
    __local uchar4 *vertices, __local real *planes,
#if P1>0
    __global real *planes2,
#endif
    __local uchar *next_boundaries, __local int *planesNeighbour
    , int num, __global unsigned int const *ids, int ids_offset
    , __global real4 const * pts, __global unsigned int const * neigs
    , __global Status* gpu_stat, __global uchar *usedCells
)
{
    int gid=get_global_id(0);
    if (gid>=num) return;
    int lid=get_local_id(0);
    ConvexCell cell;
    ConvexCell *cc=&cell;
    cc->m_vertices=&vertices[lid];
    cc->m_planes=&planes[lid];
#if P1>0
    cc->m_planes2=&planes2[4*gid*P1];
#endif
    cc->m_next_boundaries=&next_boundaries[4*lid];
    cc->m_planesNeighbour=&planesNeighbour[lid];
    int seed=(int) ids[gid+ids_offset];
    gpu_stat[seed]=success;
    if (pts[seed].w<=NO_CELL_LIMITS)
        return;
#if P1>0
    char done=0;
#endif
    CCInit(cc, seed, pts, gpu_stat+seed);
    for (int i=0; i<K; ++i) {
        unsigned int z = neigs[K * gid + i];
        int cur_v = CCNewPlane(cc,z); // add new plane equation
        CCClipByPlane(cc,cur_v);
        if (gpu_stat[seed] != success) return;
        if (CCIsSecurityRadiusReached(cc, pts[z]))  break;
#if P1>0
        if (done==0 && cc->nb_plane+1>=PL) {
            ++done;
            CCRenumberPlanes(cc);
        }
#endif
    }
    if (!CCIsSecurityRadiusReached(cc, pts[neigs[K * (gid + 1) - 1]])) {
        gpu_stat[seed] = security_radius_not_reached;
        return;
    }
    if (gpu_stat[seed] != success) {
        printf("\n[compute_bordering]\n============Invalid cell not captured============\n");
        return;
    }
    CCUpdateUsedCell(cc, usedCells, CUSED_CELL_DIST);
}
#else
// main code
__kernel void compute_voro_cell(
    __local uchar4 *vertices, __local real *planes,
#if P1>0
    __global real *planes2,
#endif
    __local uchar *next_boundaries
#ifdef NEED_PLANE_NEIGHBOURS
    , __local int *planesNeighbour
#endif
    , int num, __global unsigned int const *ids, int ids_offset
    , __global real4 const * pts, __global unsigned int const * neigs
#ifdef USE_GRID
    , __global char const *gridInDomain, __global real4 const *gridPoints, __global int const *gridTriangles, __global int const *gridTrianglesList, __global int const *gridOffsets
#endif
    , __global Status* gpu_stat, __global real4 * out_pts
#ifdef CHESSIAN
    , __global real* hessian, __global int *hessidx
#endif
#ifdef CUSED_CELL
    , __global uchar *usedCells
#endif
#ifdef DEBUG_COUNTER0
    , volatile __global int *debugCounter0
#endif
#ifdef DEBUG_COUNTER1
    , volatile __global int *debugCounter1
#endif
#ifdef DEBUG_COUNTER2
    , volatile __global int *debugCounter2
#endif
#ifdef DEBUG_COUNTER3
    , volatile __global int *debugCounter3
#endif
#ifdef DEBUG_COUNTER4
    , volatile __global int *debugCounter4
#endif
)
{
    int gid=get_global_id(0);
    if (gid>=num) return;
    // example of counters
    //#ifdef DEBUG_COUNTER
    //  atomic_add(&debugCounter[0],1);
    //#endif
    int lid=get_local_id(0);
    ConvexCell cell;
    ConvexCell *cc=&cell;
    cc->m_vertices=&vertices[lid];
    cc->m_planes=&planes[lid];
#if P1>0
    cc->m_planes2=&planes2[4*gid*P1];
#endif
    cc->m_next_boundaries=&next_boundaries[4*lid];
#ifdef NEED_PLANE_NEIGHBOURS
    cc->m_planesNeighbour=&planesNeighbour[lid];
#endif
    int seed=(int) ids[gid+ids_offset];
    gpu_stat[seed]=success;
#ifdef CHESSIAN
    hessidx[seed*H]=-1;
#endif
    if (pts[seed].w<=NO_CELL_LIMITS) {
        out_pts[seed]=pts[seed];
        return;
    }
    out_pts[seed]=make_real4(0,0,0,0);
    CCInit(cc, seed, pts, gpu_stat+seed);
#if P1>0
    char done=0;
#endif
    for (int i=0; i<K; ++i) {
        unsigned int z = neigs[K * gid + i];
        int cur_v = CCNewPlane(cc,z); // add new plane equation
        CCClipByPlane(cc,cur_v);
        if (gpu_stat[seed] != success) return;
        if (CCIsSecurityRadiusReached(cc, pts[z])) {
            DCOUNTER0(atomic_max(&debugCounter0[0],i+1);)
            DCOUNTER1(atomic_add(&debugCounter1[i+1],1);)
            break;
        }
#if P1>0
        if (done==0 && cc->nb_plane+1>=PL) {
            ++done;
            CCRenumberPlanes(cc);
        }
#endif
    }
#if defined(DEBUG_COUNTER0) || defined(DEBUG_COUNTER2)
    int maxP=cc->nb_plane;
#endif
#if defined(DEBUG_COUNTER0) || defined(DEBUG_COUNTER3)
    int maxV=cc->nb_vertex;
#endif

    if (!CCIsSecurityRadiusReached(cc, pts[neigs[K * (gid + 1) - 1]])) {
        gpu_stat[seed] = security_radius_not_reached;
        DCOUNTER0(atomic_max(&debugCounter0[1], maxP);)
        DCOUNTER2(atomic_add(&debugCounter2[maxP],1);)
        DCOUNTER0(atomic_max(&debugCounter0[2], maxV);)
        DCOUNTER3(atomic_add(&debugCounter3[maxV],1);)
        return;
    }
    if (gpu_stat[seed] != success) {
        printf("\n[compute_voro_cell_GridCPU]\n============Invalid cell not captured============\n");
        return;
    }
#ifdef USE_GRID
#if P1>0
    if (done==0)
      CCRenumberPlanes(cc);
#endif
    // STEP 2:   find bbox and intersecting triangle list
    real bb_min[3];
    real bb_max[3];
    for (int d=0; d<3; ++d) {
        bb_min[d]=1000;
        bb_max[d]=0;
    }
    for (int ve=0; ve<cc->nb_vertex; ++ve) {
        real4 p = CCComputeVertexCoordinates(cc, *CCVertex(cc, ve), true);
        real* tp = (real*) &p;
        for (int d=0; d<3; ++d) if (bb_min[d]>tp[d]) bb_min[d] = tp[d];
        for (int d=0; d<3; ++d) if (bb_max[d]<tp[d]) bb_max[d] = tp[d];
    }

    int voxel_min[3]; 
    int voxel_max[3];
    for (int d=0; d<3; ++d) voxel_min[d] = max(0, (int) (bb_min[d] / GRID_VOXEL_SIZE));
    for (int d=0; d<3; ++d) voxel_max[d] = min(GRID_SIZE-1, (int) (bb_max[d] / GRID_VOXEL_SIZE));
    
    real4 Pt0;
    char inDomain=1;

#if T > 0
    int (tr_ids[2])[T];
    int tr_wh=0;
    int tr_ids_size = 0;
#endif
    for (int z = voxel_min[2]; z<=voxel_max[2]; z++) {
        for (int y = voxel_min[1]; y<=voxel_max[1]; y++) {
            int id=gridId(0, y, z);
            for (int x = voxel_min[0]; x<=voxel_max[0]; x++) {
                if (inDomain==1 && gridInDomain[id+x]!=1) {
                    inDomain= gridInDomain[id+x];
                    Pt0=make_real4(x*GRID_VOXEL_SIZE, y*GRID_VOXEL_SIZE, z*GRID_VOXEL_SIZE, 1);
                }
                int minOffset=gridOffsets[id + x ];
                int maxOffset=gridOffsets[id + x + 1 ];
                if (minOffset == maxOffset) continue;

#  if T==0
                // first pass: don't process border
                gpu_stat[seed] = triangle_overflow;
                DCOUNTER0(atomic_max(&debugCounter0[1], maxP);)
                DCOUNTER2(atomic_add(&debugCounter2[maxP],1);)
                DCOUNTER0(atomic_max(&debugCounter0[2], maxV);)
                DCOUNTER3(atomic_add(&debugCounter3[maxV],1);)
                return;
#  else
                if (!merge(&(tr_ids[tr_wh][0]), tr_ids_size, &gridTrianglesList[minOffset], maxOffset-minOffset, &(tr_ids[1-tr_wh][0]), &tr_ids_size)) {
                    gpu_stat[seed] = triangle_overflow;
                    return;
                }
                tr_wh=1-tr_wh;
#  endif
            }
        }
    }
#if T>0
    DCOUNTER0(atomic_max(&debugCounter0[3], tr_ids_size);)
    DCOUNTER4(atomic_add(&debugCounter4[tr_ids_size],1);)
#endif
    if (inDomain==1) {
#ifdef GRID_OUT_DOMAIN
        // we need to use another corner here, check the not examined corners
        if (voxel_max[2]+1<GRID_SIZE) {
            int z=voxel_max[2]+1;
            for (int y = voxel_min[1]; y<=voxel_max[1]+1; y++) {
                if (y>=GRID_SIZE) break;
                int id=gridId(0, y, z);
                for (int x = voxel_min[0]; x<=voxel_max[0]+1; x++) {
                    if (x>=GRID_SIZE) break;
                    if (gridInDomain[id+x]!=1) {
                        inDomain= gridInDomain[id+x];
                        Pt0=make_real4(x*GRID_VOXEL_SIZE, y*GRID_VOXEL_SIZE, z*GRID_VOXEL_SIZE, 1);
                        break;
                    }
                }
                if (inDomain!=1) break;
            }
        }
        else {
            // assume border is outside the volume
            inDomain=0;
            Pt0=make_real4(voxel_min[0]*GRID_VOXEL_SIZE, voxel_min[1]*GRID_VOXEL_SIZE, (voxel_max[2]+1)*GRID_VOXEL_SIZE, 1);
        }
        if (inDomain==1 && voxel_max[1]+1<GRID_SIZE) {
            int y=voxel_max[1]+1;
            for (int z = voxel_min[2]; z<=voxel_max[2]; z++) {
                int id=gridId(0, y, z);
                for (int x = voxel_min[0]; x<=voxel_max[0]+1; x++) {
                    if (x>=GRID_SIZE) break;
                    if (gridInDomain[id+x]!=1) {
                        inDomain= gridInDomain[id+x];
                        Pt0=make_real4(x*GRID_VOXEL_SIZE, y*GRID_VOXEL_SIZE, z*GRID_VOXEL_SIZE, 1);
                        break;
                    }
                }
                if (inDomain!=1) break;
            }
        }
        else if (inDomain==1) {
            // assume border is outside the volume
            inDomain=0;
            Pt0=make_real4(voxel_min[0]*GRID_VOXEL_SIZE, (voxel_max[1]+1)*GRID_VOXEL_SIZE, voxel_min[2]*GRID_VOXEL_SIZE, 1);
        }
        if (inDomain==1 && voxel_max[0]+1<GRID_SIZE) {
            int x=voxel_max[0]+1;
            for (int z = voxel_min[2]; z<=voxel_max[2]; z++) {
                for (int y = voxel_min[1]; y<=voxel_max[1]; y++) {
                    int id=gridId(0, y, z);
                    if (gridInDomain[id+x]!=1) {
                        inDomain= gridInDomain[id+x];
                        Pt0=make_real4(x*GRID_VOXEL_SIZE, y*GRID_VOXEL_SIZE, z*GRID_VOXEL_SIZE, 1);
                        break;
                    }
                }
                if (inDomain!=1) break;
            }
        }
        else if (inDomain==1) {
            // assume border is outside the volume
            inDomain=0;
            Pt0=make_real4((voxel_max[0]+1)*GRID_VOXEL_SIZE, voxel_min[1]*GRID_VOXEL_SIZE, voxel_min[2]*GRID_VOXEL_SIZE, 1);
        }
        if (inDomain==1) {
            // ok, we are stuck
            gpu_stat[seed] = find_another_beginning_vertex;
            return;
        }
#else
        gpu_stat[seed] = find_another_beginning_vertex;
        return;
#endif
    }
#endif

#ifdef CHESSIAN
    int neighb[P];
    real neighb_intr[HC*P];
    int maxUsedP=min(cc->nb_plane+4,P);
    for (int p=0; p<maxUsedP; ++p) neighb[p]=-1;
    for (int p=0; p<HC*maxUsedP; ++p) neighb_intr[p]=0;
#endif

#ifndef USE_GRID
    bool addVolume=true;
#else
    bool addVolume=inDomain==2;
#endif

    if (addVolume) {
#ifndef CHESSIAN
        CCExportBaryVolume(cc, out_pts+seed, 1);
#else
        CCExportBaryVolumeAndArea(cc, out_pts+seed, neighb, neighb_intr, 1);
#endif
    }
    
#if defined(USE_GRID) && (T>0)
    if (tr_ids_size) {
        ConvexCellBackUp backup;
        CCBUInit(&backup, cc);
        for(int i=0; i<tr_ids_size; ++i) {
            int tri_id = tr_ids[tr_wh][i];
            real4 Pt1 = gridPoints[gridTriangles[3 * tri_id + 0]];
            real4 Pt2 = gridPoints[gridTriangles[3 * tri_id + 1]];
            real4 Pt3 = gridPoints[gridTriangles[3 * tri_id + 2]];
            bool go_out_triangle = det3x3R4(Pt1-Pt0, Pt2-Pt0, Pt3-Pt0)>0;
            if (!go_out_triangle) swapReal4(Pt2, Pt3);
            CCClipTetFromPointsInfinite(cc, Pt0, Pt1, Pt2, Pt3);
#  if defined(DEBUG_COUNTER0) || defined(DEBUG_COUNTER2)
            maxP=max(maxP, (int) cc->nb_plane);
#  endif
#  if defined(DEBUG_COUNTER0) || defined(DEBUG_COUNTER2)
            maxV=max(maxV, (int) cc->nb_vertex);
#  endif
            if (gpu_stat[seed] == empty_cell) {
                CCBURestore(&backup, cc);
                gpu_stat[seed]=success; // reset status to ok
                continue;
            }
            if (gpu_stat[seed] != success) {
                out_pts[seed].w = 0;
                return;
            }
#  ifndef CHESSIAN
            CCExportBaryVolume(cc, out_pts+seed, go_out_triangle ? 1.f : -1.f);
#  else
            CCExportBaryVolumeAndArea(cc, out_pts+seed, neighb, neighb_intr, go_out_triangle ? 1.f : -1.f);
#  endif
            CCBURestore(&backup, cc);
        }
    }
#endif

#ifdef CUSED_CELL
    if (out_pts[seed].w >= USED_CELL_EPSILON) {
        usedCells[seed]=1;
        // add 1-neighbouring cells
#  if CUSED_CELL_DIST>=1
        CCUpdateUsedCell(cc, usedCells, 1);
#  endif
    }
#endif
    
    DCOUNTER0(atomic_max(&debugCounter0[1], maxP);)
    DCOUNTER2(atomic_add(&debugCounter2[maxP],1);)
    DCOUNTER0(atomic_max(&debugCounter0[2], maxV);)
    DCOUNTER3(atomic_add(&debugCounter3[maxV],1);)

#if defined(CHESSIAN)
    int cnt = 0;
    for(int p=0; p<cc->nb_plane; ++p) {
        if (neighb[p] < 0 || neighb_intr[HC * p + HC - 1] <= 0.2) continue; // probably intr[HC*p+HC-1]<eps
        if (cnt>=H) {
            gpu_stat[seed]=hessian_overflow;
            return;
        }
#  if HC==1
        hessian[seed*H + cnt] = neighb_intr[p];
#  else
        for (int c=0; c<4; ++c) hessian[4*seed*H + 4*cnt + c] = neighb_intr[4 * p + c];
#  endif
        hessidx[seed*H + cnt] = neighb[p];
        cnt++;
    }
    if (cnt<H) hessidx[seed*H + cnt]=-1;
    DCOUNTER0(atomic_max(&debugCounter0[4],cnt);)
#endif
}
#endif
