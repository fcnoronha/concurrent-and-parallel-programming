#include <vector>
#include <regex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <pthread.h>

#define MAX_BUFF 1024 // Maximum Buffer
pthread_mutex_t lock; // Critical session

struct struct_arquivo {
    // Stores information about each file

    char path[MAX_BUFF]; // Path to each file
    std::vector<int> linhas; // Vector to store matches of the regex
};

struct struct_par {
    // Struct to deal with the data passed to each thread

    struct_arquivo *v; // Vector with files information
    const char *rgx_pesquisa;
    int *idx; // Ticket count
    int qtdArquivos;
};

#define ERRO(...) { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
}

void *pgrep(void *arg){

    // Argument of thread work
    struct_par *aux = (struct_par *)arg;

    // Reading each step
    struct_arquivo *v = aux->v;
    const char *rgx_pesquisa = aux->rgx_pesquisa;
    int *idx = aux->idx;
    int qtdArquivos = aux->qtdArquivos;

    // Making thread work until there are files to process
    while (1){

        // Critical session opening, implementing tickets
        pthread_mutex_lock(&lock);

        // There is no more files to be processed
        if (*idx >= qtdArquivos){
            pthread_mutex_unlock(&lock);
            return NULL;
        }

        // Path of the file I have to read
        char *path = v[*idx].path;

        // Future use in the function
        int actual_idx = *idx;
        (*idx)++;

        pthread_mutex_unlock(&lock);

        FILE *file = fopen(path, "r"); // Opening file
        char linha[MAX_BUFF];
        int cntLinha = 0;

        // Reading every line of the file
        while (fgets(linha, MAX_BUFF, file)) {

            std::regex rgx(rgx_pesquisa);

            // Checking for matche
            bool encontrou = std::regex_search(linha, rgx);

            // Storing the line of ocurrence of the match
            if (encontrou)
                v[actual_idx].linhas.push_back(cntLinha);

            cntLinha++;
        }
    }
}

void imprime(int n, struct_arquivo *v){
    for (int i = 0; i < n; i++)
        for (auto u : v[i].linhas)
            std::cout << v[i].path << " " << u << "\n";
}

// Recursively open the directory and subdiretories, storing information about the files
void recursiveOpenDir (const char * caminho_do_diretorio, std::vector<struct_arquivo> &arquivo) {
    // Readign stuff
    DIR *diretorio;
	DIR *diretorio_interno;
	struct dirent *pDirent;
	FILE *file;
	char buff[MAX_BUFF];

	diretorio = opendir(caminho_do_diretorio);

	while ((pDirent = readdir(diretorio)) != NULL) {

        // Not allowing loop inside of some folder
		if (!strcmp(pDirent->d_name, ".")) continue;
		if (!strcmp(pDirent->d_name, "..")) continue;

        // Geting path + filename
		strcpy(buff, caminho_do_diretorio);
        strcat(buff, "/");
		strcat(buff, pDirent->d_name);
		diretorio_interno = opendir(buff);

		if (diretorio_interno) {
			closedir(diretorio_interno);
			recursiveOpenDir (buff, arquivo);
			continue;
		}

		file = fopen(buff, "r");

        if (file == NULL)
			continue;

        // Storing information about current file
        struct_arquivo aux;
        strcpy(aux.path, buff);
        arquivo.push_back(aux);

		fclose(file);
	}

	closedir(diretorio);
}

int main(int argc, char const *argv[]) {

    std::vector<struct_arquivo> arquivo;

    // Used for ticket
    int idx = 0;

    // Lendo argumentos da linha de comando
    int MAX_THREADS = std::stoi(argv[1]);
    const char * REGEX_PESQUISA = argv[2];
    const char * CAMINHO_DO_DIRETORIO = argv[3];

    // Inittializing MUTEX
    if(pthread_mutex_init(&lock, NULL))
        ERRO("Erro ao iniciar MUTEX\n");

	recursiveOpenDir (CAMINHO_DO_DIRETORIO, arquivo);

    // Getting the total number of files
    int qtdArquivos = arquivo.size();

    // Number of threads can't be bigger than the amount of files
    MAX_THREADS = std::min(MAX_THREADS, qtdArquivos);

    // Allocating threads
    pthread_t *threads = new pthread_t[MAX_THREADS];
    struct_par *params = new struct_par;

    // Setting up parameters
    (*params).v = arquivo.data();
    (*params).rgx_pesquisa = REGEX_PESQUISA;
    (*params).idx = &idx;
    (*params).qtdArquivos = qtdArquivos;

    // Inittializing threads
    for (int i = 0; i < MAX_THREADS; i++)
        if (pthread_create(&threads[i], NULL, pgrep, (void *)params))
            ERRO("Não foi possivel inicializar as threads\n");

    // Finishing threads
    for (int i = 0; i < MAX_THREADS; ++i)
        if (pthread_join(threads[i], NULL))
            ERRO("Não foi possivel finalizar as threads\n");

    // Destroying mutex
    if (pthread_mutex_destroy(&lock))
        ERRO("Não foi possivel finalizar a MUTEX\n");

    delete[] threads;
    delete params;

    imprime(qtdArquivos, arquivo.data());

    return 0;
}
