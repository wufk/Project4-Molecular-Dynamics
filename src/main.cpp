// Project4-MolecularDynamics.cpp : Defines the entry point for the console application.
//

#include "main.h"
#include "ThreeD.h"
#include <stdio.h>

int main()
{
	ThreeD *md = new ThreeD();

	printf("default parameter or not?(Y/N)");
	char str = fgetc(stdin);
	if (str == 'N') {
		md->input();
	}
	md->init();

	printf("MD loop start\n");
	for (int i = 1; i < md->N; i++) {
		md->MDLoop(i);
		//    md->CNP();
	}
	printf("MD loop complete\n");

	delete md;
	char buf[10];
	fgets(buf, 10, stdin);
    return 0;
}

