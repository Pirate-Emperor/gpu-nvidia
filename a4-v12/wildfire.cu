// Your solution. This is the only file you submit.
// It is compiled together with the provided harness.cu:
//     nvcc -arch=sm_70 harness.cu wildfire.cu -o main
// See wildfire.h for the definitions you can use and the full contract of
// run_simulation().
#include "wildfire.h"

#define BLOCK_SIZE 256 // you can change it.

// ============================================================================
// CUDA KERNELS
// ============================================================================

#define XS 13
#define XM 8191
#define MP 32
#define IP 16384

__device__ __forceinline__ int enc(int r,int c){return (r<<XS)|c;}
__device__ __forceinline__ int er(int x){return x>>XS;}
__device__ __forceinline__ int ec(int x){return x&XM;}

__global__ void init_fire(const int *fir,int nf,uint8_t *st,int *br,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nf)return;
    int r=fir[2*i],c=fir[2*i+1];
    st[r*n+c]=BURNING;
    br[i]=enc(r,c);
}

__global__ void init_stn(const int *src,int ns,uint8_t *msk,int *stn,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ns)return;
    int r=src[2*i],c=src[2*i+1];
    msk[r*n+c]=1;
    stn[i]=enc(r,c);
}

__global__ void heat_wk(const int *br,int nb,const uint8_t *st,const uint8_t *fu,
                        const uint8_t *el,const uint8_t *msk,int *ht,int *ca,int *nc,
                        Params p)
{
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if(q>=nb)return;

    int x=br[q],sr=er(x),sc=ec(x),se=el[sr*p.N+sc];
    #pragma unroll
    for(int rr=-2;rr<=2;++rr) {
        #pragma unroll
        for(int cc=-2;cc<=2;++cc) {
            if(rr==0&&cc==0)continue;
            int tr=sr+rr,tc=sc+cc;
            if((unsigned)tr>=(unsigned)p.N||(unsigned)tc>=(unsigned)p.N)continue;
            int id=tr*p.N+tc;
            if(st[id]!=UNBURNT||fu[id]==0||msk[id])continue;

            int a=p.H,b=1;
            if((rr==p.dr&&cc==p.dc)||(rr==2*p.dr&&cc==2*p.dc))a*=p.W;
            else if((rr==-p.dr&&cc==-p.dc)||(rr==-2*p.dr&&cc==-2*p.dc))b*=p.W;
            int de=(int)el[id]-(int)se;
            if(de>p.D)a*=p.S;
            else if(de<-p.D)b*=p.S;

            int v=b==1?a:a/b;
            if(v==0)continue;
            int old=atomicAdd(&ht[id],v);
            if(old==0)ca[atomicAdd(nc,1)]=enc(tr,tc);
        }
    }
}

__global__ void heat_fg(const uint8_t *st,const uint8_t *fu,const uint8_t *el,
                        const uint8_t *msk,int *ca,int *nc,Params p)
{
    __shared__ uint8_t ss[20][36],ee[20][36];
    int tx=threadIdx.x,ty=threadIdx.y,bx=blockDim.x,by=blockDim.y;
    int rb=blockIdx.y*by-2,cb=blockIdx.x*bx-2;

    for(int y=ty;y<by+4;y+=by)
        for(int x=tx;x<bx+4;x+=bx) {
            int r=rb+y,c=cb+x;
            if((unsigned)r<(unsigned)p.N&&(unsigned)c<(unsigned)p.N) {
                int id=r*p.N+c;
                ss[y][x]=st[id];
                ee[y][x]=el[id];
            } else {
                ss[y][x]=UNBURNT;
                ee[y][x]=0;
            }
        }
    __syncthreads();

    int r=blockIdx.y*by+ty,c=blockIdx.x*bx+tx;
    if(r>=p.N||c>=p.N)return;

    int id=r*p.N+c;
    if(ss[ty+2][tx+2]!=UNBURNT||fu[id]==0||msk[id])return;

    int h=0,se=ee[ty+2][tx+2];
    #pragma unroll
    for(int rr=-2;rr<=2;++rr) {
        #pragma unroll
        for(int cc=-2;cc<=2;++cc) {
            if(rr==0&&cc==0)continue;
            int sy=ty+2+rr,sx=tx+2+cc;
            if(ss[sy][sx]!=BURNING)continue;

            int a=p.H,b=1,dr=-rr,dc=-cc;
            if((dr==p.dr&&dc==p.dc)||(dr==2*p.dr&&dc==2*p.dc))a*=p.W;
            else if((dr==-p.dr&&dc==-p.dc)||(dr==-2*p.dr&&dc==-2*p.dc))b*=p.W;

            int de=se-(int)ee[sy][sx];
            if(de>p.D)a*=p.S;
            else if(de<-p.D)b*=p.S;

            h+=b==1?a:a/b;
        }
    }
    if(h>=p.I)ca[atomicAdd(nc,1)]=enc(r,c);
}

__global__ void end_old(const int *br,int nb,uint8_t *st,uint8_t *fu,uint8_t *wt,
                        int *nx,int *nn,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nb)return;

    int x=br[i],r=er(x),c=ec(x),id=r*n+c;
    uint8_t f=(uint8_t)(fu[id]-1);
    fu[id]=f;

    if(wt[id]) {
        st[id]=WET;
        wt[id]=0;
    } else if(f==0) st[id]=BURNT;
    else {
        st[id]=BURNING;
        nx[atomicAdd(nn,1)]=x;
    }
}

__global__ void add_new(const int *ca,int nc,int *ht,uint8_t *st,const uint8_t *fu,
                        const uint8_t *msk,int *nx,int *nn,int ck,Params p)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nc)return;

    int x=ca[i],r=er(x),c=ec(x),id=r*p.N+c,h=ck?ht[id]:p.I;
    if(st[id]==UNBURNT&&fu[id]>0&&!msk[id]&&h>=p.I) {
        st[id]=BURNING;
        nx[atomicAdd(nn,1)]=x;
    }
    if(ck)ht[id]=0;
}

__device__ __forceinline__ bool bet(int a,int b,int r,int c,int R)
{
    if(a<0)return false;
    if(b<0)return true;
    int ar=er(a),ac=ec(a),br=er(b),bc=ec(b);
    int da=max(abs(ar-r),abs(ac-c));
    int db=max(abs(br-r),abs(bc-c));
    int st=R>0?R:1;
    da=(da+st-1)/st;
    db=(db+st-1)/st;
    if(da!=db)return da<db;
    return a<b;
}

// old: wet bits + clear pass.
__global__ void find_burn(const int *br,int nb,const int *cr,int *pb,
                          int np,int C,int R)
{
    int z=blockIdx.x,ci=z/np,pi=z-ci*np;
    if(ci>=C)return;

    int r=cr[3*ci],c=cr[3*ci+1],w=cr[3*ci+2];
    __shared__ int sm[BLOCK_SIZE];

    if(w==0) {
        if(threadIdx.x==0)pb[z]=-1;
        return;
    }

    int lo=(int)(((long long)pi*nb)/np);
    int hi=(int)(((long long)(pi+1)*nb)/np);
    int best=-1;
    for(int j=lo+threadIdx.x;j<hi;j+=blockDim.x)
        best=bet(best,br[j],r,c,R)?best:br[j];

    sm[threadIdx.x]=best;
    __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1) {
        if(threadIdx.x<d)
            sm[threadIdx.x]=bet(sm[threadIdx.x],sm[threadIdx.x+d],r,c,R)
                          ?sm[threadIdx.x]:sm[threadIdx.x+d];
        __syncthreads();
    }
    if(threadIdx.x==0)pb[z]=sm[0];
}

__global__ void red_burn(const int *br,int nb,const int *pb,const int *stn,int ns,
                         int *cr,uint8_t *st,const uint8_t *msk,uint8_t *wt,
                         int n,int np,int C,int R,int X)
{
    int ci=blockIdx.x;
    if(ci>=C)return;

    int r=cr[3*ci],c=cr[3*ci+1],w=cr[3*ci+2];
    __shared__ int sm[BLOCK_SIZE];
    int best=-1;

    if(w>0) {
        if(np==1) {
            for(int j=threadIdx.x;j<nb;j+=blockDim.x)
                best=bet(best,br[j],r,c,R)?best:br[j];
        } else {
            for(int j=threadIdx.x;j<np;j+=blockDim.x)
                best=bet(best,pb[ci*np+j],r,c,R)?best:pb[ci*np+j];
        }
    } else {
        for(int j=threadIdx.x;j<ns;j+=blockDim.x)
            best=bet(best,stn[j],r,c,R)?best:stn[j];
    }

    sm[threadIdx.x]=best;
    __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1) {
        if(threadIdx.x<d)
            sm[threadIdx.x]=bet(sm[threadIdx.x],sm[threadIdx.x+d],r,c,R)
                          ?sm[threadIdx.x]:sm[threadIdx.x+d];
        __syncthreads();
    }

    if(threadIdx.x==0) {
        int t=sm[0];
        if(t>=0) {
            int tr=er(t),tc=ec(t),dr=tr-r,dc=tc-c;
            dr=dr>R?R:(dr<-R?-R:dr);
            dc=dc>R?R:(dc<-R?-R:dc);
            r+=dr;c+=dc;
        }
        int id=r*n+c;
        if(w>0&&st[id]==BURNING) {
            --w;
            wt[id]=1;
        }
        if(msk[id])w=X;
        cr[3*ci]=r;
        cr[3*ci+1]=c;
        cr[3*ci+2]=w;
    }
}

// fix: -1 target ordering.
__global__ void idle_cr(const int *stn,int ns,int *cr,const uint8_t *msk,
                         int n,int R,int X,int C)
{
    int ci=blockIdx.x;
    if(ci>=C)return;

    int r=cr[3*ci],c=cr[3*ci+1],w=cr[3*ci+2];
    __shared__ int sm[BLOCK_SIZE];
    int best=-1;

    if(w==0)
        for(int j=threadIdx.x;j<ns;j+=blockDim.x)
            best=bet(best,stn[j],r,c,R)?best:stn[j];

    sm[threadIdx.x]=best;
    __syncthreads();
    for(int d=blockDim.x>>1;d;d>>=1) {
        if(threadIdx.x<d)
            sm[threadIdx.x]=bet(sm[threadIdx.x],sm[threadIdx.x+d],r,c,R)
                          ?sm[threadIdx.x]:sm[threadIdx.x+d];
        __syncthreads();
    }

    if(threadIdx.x==0) {
        int t=sm[0];
        if(t>=0) {
            int tr=er(t),tc=ec(t),dr=tr-r,dc=tc-c;
            dr=dr>R?R:(dr<-R?-R:dr);
            dc=dc>R?R:(dc<-R?-R:dc);
            r+=dr;c+=dc;
        }
        if(msk[r*n+c])w=X;
        cr[3*ci]=r;
        cr[3*ci+1]=c;
        cr[3*ci+2]=w;
    }
}

// ============================================================================
// SIMULATION DRIVER
// ============================================================================

void run_simulation(Params p, uint8_t *state, uint8_t *fuel, const uint8_t *elev,
                    int F, const int *fires, int K, const int *stations,
                    int C, int *crews)
{
    size_t cells=(size_t)p.N*p.N;
    size_t hbs=cells*sizeof(int);
    uint8_t *dst=0,*dfu=0,*del=0,*dms=0,*dwt=0;
    int *dht=0,*dfi=0,*dstp=0,*dba=0,*dbb=0,*dca=0,*dcr=0,*dpb=0;
    int *dnc=0,*dnn=0;

    cudaMalloc(&dst,cells);
    cudaMalloc(&dfu,cells);
    cudaMalloc(&del,cells);
    cudaMalloc(&dms,cells);
    cudaMalloc(&dwt,cells);
    bool hk=0;
    cudaMalloc(&dba,cells*sizeof(int));
    cudaMalloc(&dbb,cells*sizeof(int));
    cudaMalloc(&dca,cells*sizeof(int));
    cudaMalloc(&dcr,(size_t)(C?3*C:3)*sizeof(int));
    cudaMalloc(&dpb,(size_t)(C?C*MP:MP)*sizeof(int));
    cudaMalloc(&dnc,sizeof(int));
    cudaMalloc(&dnn,sizeof(int));

    cudaMemset(dst,0,cells);
    cudaMemset(dwt,0,cells);
    cudaMemcpy(dfu,fuel,cells,cudaMemcpyHostToDevice);
    cudaMemcpy(del,elev,cells,cudaMemcpyHostToDevice);
    if(C)cudaMemcpy(dcr,crews,(size_t)3*C*sizeof(int),cudaMemcpyHostToDevice);

    if(F) {
        cudaMalloc(&dfi,(size_t)2*F*sizeof(int));
        cudaMemcpy(dfi,fires,(size_t)2*F*sizeof(int),cudaMemcpyHostToDevice);
        init_fire<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(dfi,F,dst,dba,p.N);
        cudaFree(dfi);dfi=0;
    }

    cudaMemset(dms,0,cells);
    if(K) {
        cudaMalloc(&dstp,(size_t)K*sizeof(int));
        int *tmp=0;
        cudaMalloc(&tmp,(size_t)2*K*sizeof(int));
        cudaMemcpy(tmp,stations,(size_t)2*K*sizeof(int),cudaMemcpyHostToDevice);
        init_stn<<<(K+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(tmp,K,dms,dstp,p.N);
        cudaFree(tmp);
    }

    int *br=dba,*nx=dbb,nb=F;

    for(int stp=0;stp<p.T;++stp) {
        cudaMemset(dnn,0,sizeof(int));

        int nc=0;
        if(nb>0) {
            bool wk=(p.I>0&&(long long)nb*16LL<(long long)cells);
            cudaMemset(dnc,0,sizeof(int));

            if(wk) {
                if(!hk) {
                    cudaMalloc(&dht,hbs);
                    cudaMemset(dht,0,hbs);
                    hk=1;
                }
                heat_wk<<<(nb+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                    br,nb,dst,dfu,del,dms,dht,dca,dnc,p);
            } else {
                if(p.N>=64) {
                    dim3 bs(32,8);
                    dim3 gs((p.N+31)/32,(p.N+7)/8);
                    heat_fg<<<gs,bs>>>(dst,dfu,del,dms,dca,dnc,p);
                } else {
                    dim3 bs(16,16);
                    dim3 gs((p.N+15)/16,(p.N+15)/16);
                    heat_fg<<<gs,bs>>>(dst,dfu,del,dms,dca,dnc,p);
                }
            }

            if(C) {
                int np=(nb+IP-1)/IP;
                np=np<1?1:np;
                np=np>MP?MP:np;
                int lim=IP/C;
                lim=lim<1?1:lim;
                np=np<lim?np:lim;

                if(np>1)find_burn<<<C*np,BLOCK_SIZE>>>(br,nb,dcr,dpb,np,C,p.R);
                red_burn<<<C,BLOCK_SIZE>>>(
                    br,nb,dpb,dstp,K,dcr,dst,dms,dwt,p.N,np,C,p.R,p.X);
            }

            cudaMemcpy(&nc,dnc,sizeof(int),cudaMemcpyDeviceToHost);

            if(nb)
                end_old<<<(nb+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                    br,nb,dst,dfu,dwt,nx,dnn,p.N);

            if(nc)
                add_new<<<(nc+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(
                    dca,nc,dht,dst,dfu,dms,nx,dnn,wk?1:0,p);

            cudaMemcpy(&nb,dnn,sizeof(int),cudaMemcpyDeviceToHost);
            int *tmp=br;br=nx;nx=tmp;
        } else if(C) {
            idle_cr<<<C,BLOCK_SIZE>>>(dstp,K,dcr,dms,p.N,p.R,p.X,C);
            nb=0;
        }
    }

    cudaMemcpy(state,dst,cells,cudaMemcpyDeviceToHost);
    cudaMemcpy(fuel,dfu,cells,cudaMemcpyDeviceToHost);
    if(C)cudaMemcpy(crews,dcr,(size_t)3*C*sizeof(int),cudaMemcpyDeviceToHost);

    if(dfi)cudaFree(dfi);
    if(dstp)cudaFree(dstp);
    cudaFree(dst);
    cudaFree(dfu);
    cudaFree(del);
    cudaFree(dms);
    cudaFree(dwt);
    if(hk)cudaFree(dht);
    cudaFree(dba);
    cudaFree(dbb);
    cudaFree(dca);
    cudaFree(dcr);
    cudaFree(dpb);
    cudaFree(dnc);
    cudaFree(dnn);
}
