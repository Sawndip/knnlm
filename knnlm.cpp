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

int	fd;
struct	stat	sb;
uint8_t	*data;
uint64_t	data_size,	threads,	seed,	kmer;
double	mean;
uint8_t	w[1024];

uint64_t	open_mmap(const	char	*F){
	fd=open(F,	O_RDONLY);	if(fd<0)	return	0;
	fstat(fd,	&sb);
	data=(uint8_t*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
	if(data==MAP_FAILED)	return	0;
	data_size=sb.st_size;
	return	data_size;
}

void	close_mmap(void){
	munmap(data,sb.st_size);	close(fd);
}

static	inline	double	score(uint8_t	*p,	uint8_t	*q){
	size_t	s=0;	uint8_t	*a=p-kmer+1,	*b=q-kmer+1;
	for(size_t	i=0;	i<kmer;	i++)	s+=w[i]*(a[i]==b[i]);
	return	s;
}

double	predict(uint8_t	*p,	double	*prob,	double	alpha){
	double	pr[threads<<8]={};
	#pragma omp parallel for
	for(size_t	i=kmer-1;	i<data_size-1;	i++)	
		if(data[i]==*p&&(data[i-1]==*(p-1)||data[i-2]==*(p-2))&&data+i!=p){
		double	s=(score(data+i,p)-mean)*alpha;
		s=exp(s);		
		pr[(omp_get_thread_num()<<8)+data[i+1]]+=s;
	}
	memset(prob,0,256*sizeof(double));
	for(size_t	i=0;	i<threads;	i++)	for(size_t	j=0;	j<256;	j++)	prob[j]+=pr[(i<<8)+j];
	double	sp=0;
	for(size_t	i=0;	i<256;	i++)	sp+=(prob[i]+=DBL_MIN);
	sp=1/sp;
	for(size_t  i=0;    i<256;  i++)	prob[i]*=sp;
	return	-log2(fmax(prob[*(p+1)],DBL_MIN));
}

double	normalize(double	beta){
	for(size_t	i=0;	i<kmer;	i++)	w[i]=powf(beta,kmer-1-i)*255;
	seed=0;
	double	x,	sx=0,	sxx=0,	sn=0x100000;
	for(size_t	k=0;	k<0x100000;	k++){
		size_t	i=wyrand(&seed)%(data_size-kmer-2)+kmer-1,j;
		do	j=wyrand(&seed)%(data_size-kmer-2)+kmer-1;	while(j==i);
		x=score(data+i,data+j);
		sx+=x;	
		sxx+=x*x;
	}
	sx/=sn;	mean=sx;
	return	sqrt(sxx/sn-sx*sx);
}

void	document(void){
	cerr<<"usage:	knnlm [options] [word1 word2 ...]\n";
	cerr<<"\t-t:	text file=data.txt\n";
	cerr<<"\t-k:	kmer=32\n";
	cerr<<"\t-a:	sampling temperature=2.718\n";
	cerr<<"\t-b:	benckmark chars=0\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	string	file="data.txt";	double	alpha=2.718;	size_t	bench=0;	kmer=32;	
	if(ac<2)	document();
	int	opt;
	while((opt=getopt(ac,	av,	"t:k:a:b:"))>=0){
		switch(opt){
		case	't':	file=optarg;	break;
		case	'k':	kmer=atoi(optarg);	break;
		case	'a':	alpha=atof(optarg);	break;
		case	'b':	bench=atoi(optarg);	break;
		default:	document();
		}
	}
	threads=omp_get_num_procs();	omp_set_num_threads(threads);
	if(!open_mmap(file.c_str()))	return	0;
	alpha/=normalize(exp(-log(255)/kmer));
	if(!bench){
		seed=wyhash64(time(NULL),0);
		ofstream	fo("article.txt");
		vector<uint8_t>	v;
		for(size_t	i=0;	i<kmer;	i++)	v.push_back(wyrand(&seed)&255);
		for(int	i=optind;	i<ac;	i++){
			for(size_t	j=0;	j<strlen(av[i]);	j++){ v.push_back(av[i][j]); fo<<av[i][j];	}
			if(i+1<ac){	v.push_back(' ');	fo<<' ';	}
		}
		fo.flush();
		double	prob[256];
		for(;;){
			predict(v.data()+v.size()-1,prob,alpha);
			double	ran=wy2u01(wyrand(&seed)),	sum=0;
			uint8_t	c=0;
			for(size_t	i=0;	i<256;	i++){
				sum+=prob[i];
				if(sum>=ran){	c=i;	break;	}
			}
			putchar(c);	fflush(stdout);
			v.push_back(c);
			fo<<c;	fo.flush();
		}
		fo.close();
	}
	else{
		seed=0;
		cerr<<"benchmarking\n";
		timeval	beg,	end;
		gettimeofday(&beg,NULL);
		double	sx=0;
		for(size_t	i=0;	i<bench;	i++){
			double	prob[256];
			unsigned	j=wyrand(&seed)%(data_size-kmer-1)+kmer-1;
			double	l=predict(data+j,prob,alpha);
			sx+=l;
		}
		gettimeofday(&end,NULL);
		double	dt=end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec);
		cerr<<"\nloss:\t"<<sx/bench<<" bits/byte\nspeed:\t"<<bench/dt<<" chars/sec\n";

	}
	close_mmap();
	return	0;
}
