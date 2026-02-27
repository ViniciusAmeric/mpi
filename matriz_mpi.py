from mpi4py import MPI
import numpy as np


def gerar_matrizes(N, rank):

    if rank == 0:
        A = np.random.randint(0, 10, (N, N)).astype('i')
        B = np.random.randint(0, 10, (N, N)).astype('i')
        print("Matriz A:\n", A)
        print("\nMatriz B:\n", B)
    else:
        A = None
        B = None
    return A, B


def distribuir_dados(A, B, N, size, comm):
    A_local = np.zeros((N // size, N), dtype='i')
    comm.Scatter(A, A_local, root=0)

    B = comm.bcast(B, root=0)

    return A_local, B


def multiplicacao_local(A_local, B):
    """
    Cada processo multiplica sua parte local.
    """
    return np.dot(A_local, B)


def reunir_resultados(C_local, N, size, rank, comm):

    if rank == 0:
        C = np.zeros((N, N), dtype='i')
    else:
        C = None

    comm.Gather(C_local, C, root=0)
    return C


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 🔒 Tamanho fixo seguro
    N = 4

    # 🔒 Permitimos apenas 1, 2 ou 4 processos
    allowed_processes = [1, 2, 4]

    if size not in allowed_processes:
        if rank == 0:
            print("Erro: Execute com 1, 2 ou 4 processos.")
            print("Exemplo: mpirun -np 4 python matriz_mpi.py")
        return

    A, B = gerar_matrizes(N, rank)

    comm.Barrier()
    start = MPI.Wtime()

    A_local, B = distribuir_dados(A, B, N, size, comm)
    C_local = multiplicacao_local(A_local, B)
    C = reunir_resultados(C_local, N, size, rank, comm)

    comm.Barrier()
    end = MPI.Wtime()

    if rank == 0:
        print("\nResultado C = A x B:\n", C)
        print(f"\nTempo paralelo: {end - start:.6f} segundos")


if __name__ == "__main__":
    main()