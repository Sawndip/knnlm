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
uint64_t	data_size,	threads,	seed;
double	w[kmer];

uint64_t	open_mmap(const	char	*F){
	fd=open(F,	O_RDONLY);	if(fd<0)	return	0;
	fstat(fd,	&sb);
	data=(uint8_t*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
	if(data==MAP_FAILED)	return	0;
	data_size=sb.st_size;
	return	wyhash(data,data_size,0,_wyp);
}

void	close_mmap(void){
	munmap(data,sb.st_size);	close(fd);
}

static	inline	double	score(uint8_t	*p,	uint8_t	*q){
	double	s=0;	uint8_t	*a=p-kmer+1,	*b=q-kmer+1;
	for(size_t	i=0;	i<kmer;	i++)	s+=w[i]*(a[i]==b[i]);
	return	s;
}

double	predict(uint8_t	*p,	double	*prob){
	double	pr[threads<<8]={};
	#pragma omp parallel for
	for(size_t	i=kmer-1;	i<data_size-1;	i++){
		v8qi	r=__builtin_ia32_pcmpeqb(*(v8qi*)(data+i-7),*(v8qi*)(p-7));
		if(__builtin_popcountll(*(uint64_t*)&r)>=16&&data+i!=p){
		size_t	tid=omp_get_thread_num();
		double	s=score(data+i,p);
		pr[(tid<<8)+data[i+1]]+=exp(s);
		}
	}
	memset(prob,0,256*sizeof(double));
	for(size_t	i=0;	i<threads;	i++)	for(size_t	j=0;	j<256;	j++)	prob[j]+=pr[(i<<8)+j];
	double	sp=0;
	for(size_t	i=0;	i<256;	i++)	sp+=(prob[i]+=FLT_MIN);
	sp=1/sp;
	for(size_t  i=0;    i<256;  i++)	prob[i]*=sp;
	return	-log2(prob[p[1]]);
}

void	document(void){
	cerr<<"usage:	knnlm [options] text [word1 word2 ...]\n";
	cerr<<"\t-b:	benckmark chars=0\n";
	cerr<<"\t-t:	threads=0\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	size_t	bench=0;	threads=omp_get_num_procs();
	if(ac<2)	document();
	int	opt;
	while((opt=getopt(ac,	av,	"b:t:"))>=0){
		switch(opt){
		case	'b':	bench=atoi(optarg);	break;
		case	't':	threads=atoi(optarg);	break;
		default:	document();
		}
	}
	omp_set_num_threads(threads);
	if(!open_mmap(av[optind]))	return	0;
	ifstream	fi("model");
	for(size_t	i=0;	i<kmer;	i++)	fi>>w[i];
	fi.close();
	if(!bench){
		seed=wyhash64(time(NULL),0);		
		vector<uint8_t>	v;
		for(size_t	i=0;	i<kmer;	i++)	v.push_back(wyrand(&seed)&255);
		for(int	i=optind+1;	i<ac;	i++){
			for(size_t	j=0;	j<strlen(av[i]);	j++)	v.push_back(av[i][j]);
			if(i+1<ac)	v.push_back(' ');
		}
		double	prob[256];
		for(;;){
			predict(v.data()+v.size()-1,prob);
			double	ran=wy2u01(wyrand(&seed)),	sum=0;
			uint8_t	c=0;
			for(size_t	i=0;	i<256;	i++){
				sum+=prob[i];
				if(sum>=ran){	c=i;	break;	}
			}
			putchar(c);	fflush(stdout);
			v.push_back(c);
		}
	}
	else{
		seed=0;
		cerr<<"benchmarking\n";
		timeval	beg,	end;
		gettimeofday(&beg,NULL);
		double	sx=0;
		for(size_t	i=0;	i<bench;	i++){
			double	prob[256];
			unsigned	j=wyrand(&seed)%(data_size-kmer)+kmer-1;
			double	l=predict(data+j,prob);
			sx+=l;
		}
		gettimeofday(&end,NULL);
		double	dt=end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec);
		cerr<<"\nloss:\t"<<sx/bench<<" bits/byte\nspeed:\t"<<bench/dt<<" chars/sec\n";
	}
	close_mmap();
	return	0;
}
