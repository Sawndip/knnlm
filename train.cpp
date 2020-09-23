#include	<x86intrin.h>
#include	<sys/time.h>
#include	<sys/mman.h>
#include	<sys/stat.h>
#include	<algorithm>
#include	<unistd.h>
#include	<iostream>
#include	"wyhash.h"
#include	<fcntl.h>
#include	<fstream>
#include	<cstdlib>
#include	<fstream>
#include	<cfloat>
#include	<vector>
#include	<cmath>
#include	<omp.h>
using	namespace	std;
const	unsigned	kmer=128;
typedef    char   v8qi    __attribute__   ((__vector_size__       (8)));

int	fd;
struct	stat	sb;
uint8_t	*data;
uint64_t	data_size,	threads=omp_get_num_procs(),	seed=wyhash64(time(NULL),0);
double	a[kmer]={};

uint64_t	open_mmap(const	char	*F){
	fd=open(F,	O_RDONLY);	if(fd<0)	return	0;
	fstat(fd,	&sb);
	data=(uint8_t*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
	if(data==MAP_FAILED)	return	0;
	data_size=sb.st_size;
	if(data_size>(1ull<<25))	data_size=(1ull<<25);
	return	wyhash(data,data_size,0,_wyp);
}

void	close_mmap(void){
	munmap(data,sb.st_size);	close(fd);
}

void	sgd(uint8_t	*p,	double	eta){
	double	sw=FLT_MIN,	sy=FLT_MIN,	sx[kmer]={},	sxy[kmer]={};
	double	vsw[threads]={},	vsy[threads]={},	vsx[threads][kmer]={},	vsxy[threads][kmer]={};
	#pragma omp parallel for
	for(size_t	i=kmer-1;	i<data_size-1;	i++){
	//	if(__builtin_popcountll(*(uint64_t*)&__builtin_ia32_pcmpeqb(*(v8qi*)(data+i-7),*(v8qi*)(p-7)))>=16&&data+i!=p){
		v8qi	r=__builtin_ia32_pcmpeqb(*(v8qi*)(data+i-7),*(v8qi*)(p-7));
		if(__builtin_popcountll(*(uint64_t*)&r)>=16&&data+i!=p){
		size_t	t=omp_get_thread_num();
		double	w=0;
		uint8_t	*m=data+i-(kmer-1),	*n=p-(kmer-1);
		for(size_t	j=0;	j<kmer;	j++)	w+=a[j]*(m[j]==n[j]);
		w=exp(w);
		double	y=data[i+1]==p[1];
		vsw[t]+=w;	vsy[t]+=w*y;
		for(size_t  j=0;    j<kmer; j++){
			double	x=(m[j]==n[j]);
			vsx[t][j]+=w*x;
			vsxy[t][j]+=w*x*y;
		}
		}
	}
	for(size_t	t=0;	t<threads;	t++){
		sw+=vsw[t];	sy+=vsy[t];
		for(size_t  j=0;    j<kmer; j++){	sx[j]+=vsx[t][j];	sxy[j]+=vsxy[t][j];	}
	}
	for(size_t  j=0;    j<kmer; j++){
		a[j]+=eta*(sxy[j]/sy-sx[j]/sw);
		if(a[j]<0)	a[j]=0;
	}
}

void	document(void){
	cerr<<"usage:	train [options] text [word1 word2 ...]\n";
	cerr<<"\t-e:	eta=0.3\n";
	cerr<<"\t-n:	step=10000\n";
	cerr<<"\t-t:	threads=cpus\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	double	eta=0.3;	size_t	step=10000;
	if(ac<2)	document();
	int	opt;
	while((opt=getopt(ac,	av,	"e:n:t:"))>=0){
		switch(opt){
		case	'e':	eta=atof(optarg);	break;
		case	'n':	step=atoi(optarg);	break;
		case	't':	threads=atoi(optarg);	break;
		default:	document();
		}
	}
	omp_set_num_threads(threads);
	if(!open_mmap(av[optind]))	return	0;
	for(size_t	j=0;	j<step;	j++){	sgd(data+wyrand(&seed)%(data_size-kmer)+kmer-1,eta);	if(j%100==99)	cerr<<'=';	}
	cerr<<'\n';	eta/=10;
	for(size_t	j=0;	j<step;	j++){	sgd(data+wyrand(&seed)%(data_size-kmer)+kmer-1,eta);	if(j%100==99) cerr<<'=';  }
	cerr<<'\n';	eta/=10;
	for(size_t	j=0;	j<step;	j++){	sgd(data+wyrand(&seed)%(data_size-kmer)+kmer-1,eta);	if(j%100==99) cerr<<'=';  }
	cerr<<'\n';
	close_mmap();
	ofstream	fo("model");
	for(size_t	j=0;	j<kmer;	j++)	fo<<a[j]<<'\n';
	fo.close();
	return	0;
}
