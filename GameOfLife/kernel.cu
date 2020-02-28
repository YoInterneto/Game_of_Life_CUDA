
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "./common/book.h"

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif


__global__ void movimientoCelular(char* matriz, char* matrizResultado, int fila, int columna);

cudaError_t lanzarKernel(char* matriz, char* matrizResultado, int fila, int columna);

int contarVivas(char* matriz, int dimension);

void imprimirMatriz(char* matriz, int dimension, int columna);

void rellenarMatriz(char* matriz, int dimension);

int main(int arg, char* argv[])
{
 
    //Comprueba que haya solo el numero de argumento permitidos
    if (arg != 4){
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
        else if(propiedades.maxThreadsPerBlock < dimension){
            printf("\nERROR: Numero de bloques supera el maximo permitido por su tarjeta.\n");
        }
        //Una vez comprobado todo empezamos con la ejecucion
        else {

            printf("\n[Matriz(%dx%d) Dimension(%d)] [modo: %s] \n", fila, columna, dimension, argv[1]);

            if (strcmp("-m", argv[1]) == 0) {
                manual = true;
            }

            //Rellenamos el tablero con celulas muertas y vivas
            rellenarMatriz(matriz,dimension);

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

                generaciones ++;
            }
        }

        //Liberamos los arrays
        free(matriz);
        free(matrizResultado);

    }
}

__global__ void movimientoCelular(char* matriz, char* matrizResultado, int fila, int columna) {

    int posicion = threadIdx.x * columna + threadIdx.y;

    int contador = 0;

    //****************************
    //Primera fila 0x
    if (threadIdx.x == 0) {
        //Posicion esquina ariba izquierda 0x0
        if (threadIdx.y == 0) {

            if ((matriz[posicion + 1]) == 'X') { contador++; }
            if ((matriz[posicion + columna]) == 'X') { contador++; }
            if ((matriz[posicion + (columna + 1)]) == 'X') { contador++; }

            //VIVA
            if (matriz[posicion] == 'X') {

                if (contador >= 2) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
            //MUERTA
            else {

                if (contador >= 3) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
        }
        //Posicion esquina superior derecha
        else if (threadIdx.y == (columna - 1)) {

            if ((matriz[posicion - 1]) == 'X') { contador++; }
            if ((matriz[posicion + columna]) == 'X') { contador++; }
            if ((matriz[posicion + (columna - 1)]) == 'X') { contador++; }

            //VIVA
            if (matriz[posicion] == 'X') {

                if (contador >= 2) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
            //MUERTA
            else {

                if (contador >= 3) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
        }
        //Posicion en la primera fila sin contar esquinas
        else {

            if ((matriz[posicion - 1]) == 'X') { contador++; }
            if ((matriz[posicion + 1]) == 'X') { contador++; }
            if ((matriz[posicion + columna]) == 'X') { contador++; }
            if ((matriz[posicion + (columna - 1)]) == 'X') { contador++; }
            if ((matriz[posicion + (columna + 1)]) == 'X') { contador++; }

            //VIVA
            if (matriz[posicion] == 'X') {

                if (contador >= 2) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
            //MUERTA
            else {

                if (contador >= 3) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
        }
    }
    //****************************
    //Ulima fila finalXx
    else if (threadIdx.x == (fila - 1)) {
        //Posicion esquina abajo izquierda
        if (threadIdx.y == 0) {

            if ((matriz[posicion + 1]) == 'X') { contador++; }
            if ((matriz[posicion - columna]) == 'X') { contador++; }
            if ((matriz[posicion - (columna - 1)]) == 'X') { contador++; }

            //VIVA
            if (matriz[posicion] == 'X') {

                if (contador >= 2) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
            //MUERTA
            else {

                if (contador >= 3) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
        }
        //Posicion esquina abajo derecha
        else if (threadIdx.y == (columna - 1)) {

            if ((matriz[posicion - 1]) == 'X') { contador++; }
            if ((matriz[posicion - columna]) == 'X') { contador++; }
            if ((matriz[posicion - (columna + 1)]) == 'X') { contador++; }

            //VIVA
            if (matriz[posicion] == 'X') {

                if (contador >= 2) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
            //MUERTA
            else {

                if (contador >= 3) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
        }
        //Posiciones ultima fila entre esquinas
        else {

            if ((matriz[posicion - 1]) == 'X') { contador++; }
            if ((matriz[posicion + 1]) == 'X') { contador++; }
            if ((matriz[posicion - columna]) == 'X') { contador++; }
            if ((matriz[posicion - (columna + 1)]) == 'X') { contador++; }
            if ((matriz[posicion - (columna - 1)]) == 'X') { contador++; }

            //VIVA
            if (matriz[posicion] == 'X') {

                if (contador >= 2) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
            //MUERTA
            else {

                if (contador >= 3) { matrizResultado[posicion] = 'X'; }
                else { matrizResultado[posicion] = 'O'; }
            }
        }
    }
    //****************************
    //Primera columna entre las dos esquinas izquierdas
    else if (threadIdx.y == 0) {

        if ((matriz[posicion + 1]) == 'X') { contador++; }
        if ((matriz[posicion - columna]) == 'X') { contador++; }
        if ((matriz[posicion + columna]) == 'X') { contador++; }
        if ((matriz[posicion + (columna + 1)]) == 'X') { contador++; }
        if ((matriz[posicion - (columna - 1)]) == 'X') { contador++; }

        //VIVA
        if (matriz[posicion] == 'X') {

            if (contador >= 2) { matrizResultado[posicion] = 'X'; }
            else { matrizResultado[posicion] = 'O'; }
        }
        //MUERTA
        else {

            if (contador >= 3) { matrizResultado[posicion] = 'X'; }
            else { matrizResultado[posicion] = 'O'; }
        }
    }
    //****************************
    //Ultima colunmna xfinalY
    else if (threadIdx.y == columna - 1) {

        if ((matriz[posicion - 1]) == 'X') { contador++; }
        if ((matriz[posicion + columna]) == 'X') { contador++; }
        if ((matriz[posicion - columna]) == 'X') { contador++; }
        if ((matriz[posicion - (columna + 1)]) == 'X') { contador++; }
        if ((matriz[posicion + (columna - 1)]) == 'X') { contador++; }

        //VIVA
        if (matriz[posicion] == 'X') {

            if (contador >= 2) { matrizResultado[posicion] = 'X'; }
            else { matrizResultado[posicion] = 'O'; }
        }
        //MUERTA
        else {

            if (contador >= 3) { matrizResultado[posicion] = 'X'; }
            else { matrizResultado[posicion] = 'O'; }
        }
    }
    //****************************
    //Posiciones fuera de los margenes
    else {

        if ((matriz[posicion + 1]) == 'X') { contador++; }
        if ((matriz[posicion - 1]) == 'X') { contador++; }
        if ((matriz[posicion + columna]) == 'X') { contador++; }
        if ((matriz[posicion - columna]) == 'X') { contador++; }
        if ((matriz[posicion - (columna + 1)]) == 'X') { contador++; }
        if ((matriz[posicion - (columna - 1)]) == 'X') { contador++; }
        if ((matriz[posicion + (columna + 1)]) == 'X') { contador++; }
        if ((matriz[posicion + (columna - 1)]) == 'X') { contador++; }

        //VIVA
        if (matriz[posicion] == 'X') {

            if (contador >= 2) { matrizResultado[posicion] = 'X'; }
            else { matrizResultado[posicion] = 'O'; }
        }
        //MUERTA
        else {

            if (contador >= 3) { matrizResultado[posicion] = 'X'; }
            else { matrizResultado[posicion] = 'O'; }
        }
    }

    //printf("CONTADOR %d: %d\n", posicion, contador);
}

cudaError_t lanzarKernel(char* matriz, char* matrizResultado, int fila, int columna) {

    char* matriz_d = NULL;
    char* matrizResultado_d = NULL;

    int dimension = fila * columna;

    cudaError_t cudaStatus;

    //Dimensiones del bloque
    dim3 blockDim(fila, columna); //Para ver arriba o abajo tendremos que sumarle el valor de la columna-1 o columna+1
                                 //Matriz(23x24)
                                 //ThreadIdx.x sera la fila (23)
                                 //ThreadIdx.y sera la columna (24)

    //Seleccionamos el device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice fallo: Tienes una GPU instalada?");
        goto Error;
    }

    //Reservamos las memorias
    cudaStatus = cudaMalloc((void**)&matriz_d , dimension * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc matriz_d fallo.");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&matrizResultado_d, dimension * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc matrizResultado_d fallo.");
        goto Error;
    }

    //Copiamos los vectores que entran por parametro
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
    movimientoCelular << < 1, blockDim >> > (matriz_d,matrizResultado_d,fila,columna);

    
    //Miramos los errores al lanzar el kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ERROR: lanzamiento de kernel fallo: %s\n",cudaGetErrorString(cudaStatus));
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
    
    for (int i = 0; i < dimension; i ++) {

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

    for (int i = 0; i < dimension; i ++) {
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


