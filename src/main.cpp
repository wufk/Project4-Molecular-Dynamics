// Project4-MolecularDynamics.cpp : Defines the entry point for the console application.
//

#include "main.h"
#include "ThreeD.h"
#include "md.h"
#include <stdio.h>

int main()
{
	ThreeD *md = new ThreeD();

	//printf("default parameter or not?(y/n)");
	//char str = fgetc(stdin);
	////if (str == 'n') {
	////	md->input();
	////}
	//md->init();

	//printf("CPU loop start\n");
	//for (int i = 1; i < md->N; i++) {
	//	md->MDLoop(i);
	//	//    md->cnp();
	//}
	//printf("CPU loop complete\n");
	//delete md;


	printf("GPU start\n");
	MD::MD_init();
	MD::MD_run();
	printf("GPU end\n");


	char buf[10];
	fgets(buf, 10, stdin);
    return 0;
}

