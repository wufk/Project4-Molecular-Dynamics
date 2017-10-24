// Project4-MolecularDynamics.cpp : Defines the entry point for the console application.
//

#include "main.h"
#include "ThreeD.h"
#include "md.h"
#include <stdio.h>
#include <ctime>

int main()
{
	printf("default parameter or not?(y/n)");
	char str[3] = { 0 };
	fgets(str, 3, stdin);
	int s = atoi(str);
	printf("%d\n", s);

	ThreeD *md = new ThreeD(s);


	//if (str == 'n') {
	//	md->input();
	//}
	md->init();

	printf("CPU loop start\n");
	
	for (int i = 1; i < md->N; i++) {
		clock_t begin = clock();
		md->MDLoop(i);
		clock_t end = clock();
		printf("time run: %f\n", (float)(begin - end));
		//    md->cnp();
	}
	
	printf("CPU loop complete\n");
	delete md;


	printf("GPU start\n");
	MD::MD_init(s, 8);
	MD::MD_run();
	printf("GPU end\n");


	char buf[10];
	fgets(buf, 10, stdin);
    return 0;
}

