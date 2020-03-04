
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "./common/book.h"

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define TILE_WIDTHx 4
#define TILE_WIDTHy 4

cudaError_t lanzarKernel(char* matriz, char* matrizResultado, int fila, int columna);

__global__ void movimientoCelularCompartida(char* matriz, char* matrizResultado, int fila, int columna);

int contarVivas(char* matriz, int dimension);

void imprimirMatriz(char* matriz, int dimension, int columna);

void rellenarMatriz(char* matriz, int dimension);

int main(int arg, char* argv[])
{

    //Comprueba que haya solo el numero de argumento permitidos
    if (arg != 4) {
        printf("\nERROR: El numero de argumentos es erróneo (.exe <-a/-m> <fila> <columna>)\n");
    }
    else {

        //Conversion de argumentos a int
        char* filaPuntero = argv[2];
        int fila = atoi(filaPuntero);
        char* columnaPuntero = argv[3];
        int columna = atoi(columnaPuntero);

        //Inicializamos cudaDeviceProp para coger las propiedades de la tarjeta
        cudaDeviceProp propiedades;
        HANDLE_ERROR(cudaGetDeviceProperties(&propiedades, 0));

        //Dimension de la matriz
        int dimension = columna * fila;

        //Matrices
        char* matriz = NULL;
        char* matrizResultado = NULL;

        matriz = (char*)malloc(sizeof(char) * dimension);
        matrizResultado = (char*)malloc(sizeof(char) * dimension);

        //Booleano para saber si el usuario quiere manual o automatico, por defecto automatico
        bool manual = false;

        //Comprueba que los numeros de columna y fila son correctos
        if (columna <= 0 | fila <= 0) {
            printf("\nERROR: La fila/columna tiene que ser un entero positivo.\n");
        }
        //Comprueba que se haya introducido el parametro de ejecucion correcto 
        else if ((strcmp("-m", argv[1]) & strcmp("-a", argv[1])) != 0) {
            printf("\nERROR: Argumentos validos solo -m[manual] o -a[automatico]\n");
        }
        else if (propiedades.maxThreadsPerBlock < dimension) {
            printf("\nERROR: Numero de bloques supera el maximo permitido por su tarjeta.\n");
        }
        //Una vez comprobado todo empezamos con la ejecucion
        else {

            printf("\n[Matriz(%dx%d) Dimension(%d)] [modo: %s] \n", fila, columna, dimension, argv[1]);

            if (strcmp("-m", argv[1]) == 0) {
                manual = true;
            }

            //Rellenamos el tablero con celulas muertas y vivas
            rellenarMatriz(matriz, dimension);

            printf("\n***TABLERO INICIAL***\n");
            imprimirMatriz(matriz, dimension, columna);

            int generaciones = 1; //Cuenta cuantas iteraciones (generaciones) han habido
            int vivas = 0;

            //Se podria poner el resultado de una funcion que cambiara el valor de un bool terminado que dijera cuando no quedan
            //mas celulas vivas por ejemplo bool terminado = false ... while(!terminado) ... if(terminar(matriz)) terminado = true
            while (vivas != dimension) {

                system("CLS");

                if (generaciones == 1) {
                    lanzarKernel(matriz, matrizResultado, fila, columna);
                }
                else {
                    lanzarKernel(matrizResultado, matrizResultado, fila, columna);
                }

                vivas = contarVivas(matrizResultado, dimension);

                printf("\nGeneracion: %d\n", generaciones);
                printf("Celulas vivas: %d\n", vivas);
                imprimirMatriz(matrizResultado, dimension, columna);

                //Si el usuario marca como manual, cada generacion tendra que pulsar alguna tecla para continuar
                if (manual) {
                    system("pause");
                }
                else {
                    Sleep(1000);
                }

                generaciones++;
            }
        }

        //Liberamos los arrays
        free(matriz);
        free(matrizResultado);

    }
}

__global__ void movimientoCelularCompartida(char* matriz, char* matrizResultado, int fila, int columna) {

    __shared__ char matrizShared[TILE_WIDTHx][TILE_WIDTHy];
    __shared__ char matrizResShared[TILE_WIDTHx][TILE_WIDTHy];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int filaPos = bx * TILE_WIDTHx + tx;
    int columnaPos = by * TILE_WIDTHy + ty;


}

cudaError_t lanzarKernel(char* matriz, char* matrizResultado, int fila, int columna) {

    //Punteros a las matrices que se meten por el kernel
    char* matriz_d;
    char* matrizResultado_d;

    int dimension = fila * columna;

    cudaError_t cudaStatus;

    //Dimension del bloque y grid
    dim3 dimGrid(fila / TILE_WIDTHx, columna / TILE_WIDTHy);
    dim3 dimBlock(TILE_WIDTHx, TILE_WIDTHy);

    //Seleccionamos el device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice fallo: Tienes una GPU instalada?");
        goto Error;
    }

    //Reservamos las memorias
    cudaStatus = cudaMalloc((void**)&matriz_d, dimension * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc matriz_d fallo.");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&matrizResultado_d, dimension * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc matrizResultado_d fallo.");
        goto Error;
    }

    //Copiamos las matrices que entran por parametro
    cudaStatus = cudaMemcpy(matriz_d, matriz, dimension * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy matriz a matriz_d fallo.");
        goto Error;
    }

    cudaStatus = cudaMemcpy(matrizResultado_d, matrizResultado, dimension * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy matrizResultado a matrizResultado_d fallo.");
        goto Error;
    }


    //Lanzamos el kernel
    movimientoCelularCompartida << < dimGrid, dimBlock >> > (matriz_d, matrizResultado_d, fila, columna);


    //Miramos los errores al lanzar el kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: lanzamiento de kernel fallo: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    //Miramos errores despues de lanzar el kernel
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: el kernel fallo con codigo %d\n", cudaStatus);
        goto Error;
    }

    //Copiamos el resultado en nuestra matriz
    cudaStatus = cudaMemcpy(matrizResultado, matrizResultado_d, dimension * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy matrizResultado_d a matrizResultado fallo.");
        goto Error;
    }


Error:
    cudaFree(matriz_d);
    cudaFree(matrizResultado_d);

    return cudaStatus;
}

void imprimirMatriz(char* matriz, int dimension, int columna) {

    for (int i = 0; i < dimension; i++) {

        if (matriz[i] == 'X') {
            printf(" 0 ");
        }
        else {
            printf(" . ");
        }

        if ((i + 1) % columna == 0) {
            printf("\n");
        }
    }
}

int contarVivas(char* matriz, int dimension) {

    int contador = 0;

    for (int i = 0; i < dimension; i++) {
        if (matriz[i] == 'X') {
            contador++;
        }
    }

    return contador;
}

void rellenarMatriz(char* matriz, int dimension) {

    srand(time(0));

    for (int i = 0; i < dimension; i++) {

        char* celula = matriz + i;

        int random = rand() % dimension + 1;

        if (random % 3 == 0 && random % 2 == 0 && random % 5 == 0) {

            *celula = 'X';
        }
        else {
            *celula = 'O';
        }

    }
}