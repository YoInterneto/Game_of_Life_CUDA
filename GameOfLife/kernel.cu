
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define FILA 32
#define COLUMNA 32 
#define DIMENSION FILA*COLUMNA

__global__ void addKernel(){

}

int main(int* arg, char* argv[])
{
    int fila;
    int columna;
    int argumentos;
    char* p;

    fila = strtol(argv[2],&p,10);
    columna = strtol(argv[3], &p, 10);


    int dimension = columna * fila;

    printf("DIMENSION (%dx%d) -> %p ", fila,columna,arg);

    bool manual = false;
    
    if (*arg != 4){
        printf("El numero de argumentos es erróneo (.exe <-a/-m> <fila> <columna>)");
    }
    else{
        if (strcmp("-m", argv[1]) == 0) {
            manual = true;
        }

        char* matriz_d;
        char* matrizRes_d;

        printf("%s,%s\n", argv[1], argv[2]);

        for (int i = 0; i < dimension; i++) {
            int random = rand() % dimension + 1;
            if (random % 10 == 0) {
                printf("X");
            }
            else {
                printf("O");
            }

            if (manual) {
                system("pause");
            }
        }
    }
   
    return 0;
}


