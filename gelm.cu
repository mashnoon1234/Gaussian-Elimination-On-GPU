#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

void rowOperation(float** __restrict__ coefMatrix, float* __restrict__ constMatrix, int row1, int row2, int size)
{
    float scalar = coefMatrix[row1][row2] / coefMatrix[row2][row2];
    for(int i = 0; i < size; i++)
    {
        coefMatrix[row1][i] -= coefMatrix[row2][i] * scalar;
    }
    constMatrix[row1] -= constMatrix[row2] * scalar;
}

bool gaussElimination(float** __restrict__ coefMatrix, float* __restrict__ constMatrix, int size)
{
    for(int row = 0; row < size; row++)
    {
        for(int rowAfterLeadingOne = row + 1; rowAfterLeadingOne < size; rowAfterLeadingOne++)
        {
            rowOperation(coefMatrix, constMatrix, rowAfterLeadingOne, row, size);
        }
    }
    return true;
}



void PrintResult(float** __restrict__ coefMatrix, float* __restrict__ constMatrix, int size)
{
    float result[size];

    result[size - 1] = constMatrix[size - 1] / coefMatrix[size - 1][size - 1];

    for(int i = size - 2; i >= 0; i--)
    {
        result[i] = constMatrix[i];
        for (int j = i + 1; j < size; j++)
            result[i] -= coefMatrix[i][j] * result[j];
        result[i] /= coefMatrix[i][i];
    }

    cout << "Result : (";

    for (int i = 0; i < size; i++)
    {
        cout << result[i];

        if(i < size - 1)
            cout << ", ";
    }

    cout << ")" << endl;

}


void CopyMatrixFromFile(float* matrix, ifstream& inFile, int size)
{
    for(int i = 0; i < size; i++)
        inFile >> matrix[i];
}

void PrintMatrix(float* matrix, int sizeOfMatrix, int size)
{
    cout << endl;
    int modValue = sizeOfMatrix;

    for(int i = 0; i < size; i++)
    {
        cout << setprecision(3) << matrix[i] << " \t";
        if(i % modValue == 0 && i != 0)
        {
        	modValue += (sizeOfMatrix + 1);
			cout << endl << endl << endl;
        }
    }
}

__global__ void gelm(float* __restrict__ d_matrix, float* __restrict__ d_result, const int sizeOfMatrix, const int size)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < size)
    {
        const int sizeOfMatrixPlus1 = sizeOfMatrix + 1;
        const int sizeOfMatrixPlus2 = sizeOfMatrixPlus1 + 1;
        const int sizeOfMatrixMinus1 = sizeOfMatrix - 1;
        const int sizeOfMatrixPlus2IntoIndex = sizeOfMatrixPlus2 * index;
        int rowIntoSizeOfMatrixPlus1PlusPointer;
        int lead;
        int i;
        for(i = 0; i < size; i += sizeOfMatrixPlus2)
        {
            if(index > (i + sizeOfMatrix))
            {
                rowIntoSizeOfMatrixPlus1PlusPointer = ((index / sizeOfMatrixPlus1) * sizeOfMatrixPlus1) + (i / sizeOfMatrixPlus1);
                d_matrix[index] -= d_matrix[index + i - rowIntoSizeOfMatrixPlus1PlusPointer] * (d_matrix[rowIntoSizeOfMatrixPlus1PlusPointer] / d_matrix[i]);
            }
        }
        __syncthreads();
        // INITIALIZATION
        if(index < sizeOfMatrix)
        {
            lead = (size - 1) - (sizeOfMatrixPlus1 * (sizeOfMatrixMinus1 - index));
            d_result[index] = d_matrix[lead] / d_matrix[lead - sizeOfMatrix + index];
			__syncthreads();
			// BACK SUBSTITUTION PROCESS
			for(i = sizeOfMatrixMinus1; i > 0; i--)
			{
				if(index < i)
					d_result[index] -= ((d_matrix[sizeOfMatrixPlus2IntoIndex + (i - index)] / d_matrix[sizeOfMatrixPlus2IntoIndex]) * d_result[i]);
			}
			__syncthreads();
        }
    }
}

int main()
{
    float* matrix;
    float* d_matrix;
    float* result;
    float* d_result;

    int sizeOfMatrix;
    ifstream inFile("inputMatrix.txt");
    inFile >> sizeOfMatrix;

    int size = sizeOfMatrix * (sizeOfMatrix + 1);

    float sizeInBytes = size * sizeof(float);

    matrix = new float [size];
    result = new float [sizeOfMatrix];

    CopyMatrixFromFile(matrix, inFile, size);

    float** coefMatrix;
    float* constMatrix;
    coefMatrix = new float* [sizeOfMatrix];
    constMatrix = new float [sizeOfMatrix];

    for(int i = 0; i < sizeOfMatrix; i++)
        coefMatrix[i] = new float [sizeOfMatrix];

    for(int row = 0; row < sizeOfMatrix; row++)
    {
        for(int column = 0; column < sizeOfMatrix; column++)
        {
            coefMatrix[row][column] = matrix[column + row * (sizeOfMatrix + 1)];

            if(column == sizeOfMatrix - 1)
                constMatrix[row] = matrix[(column + 1) + row * (sizeOfMatrix + 1)];
        }
    }

    float start_s = clock();

    bool uniqueSoln = gaussElimination(coefMatrix, constMatrix, sizeOfMatrix);
    if(uniqueSoln)
    	PrintResult(coefMatrix, constMatrix, sizeOfMatrix);

    float stop_s = clock();

    float cpu = (stop_s-start_s)/float(CLOCKS_PER_SEC)*1000;
    cout << "time CPU: " << cpu << endl;

    cudaMalloc(&d_matrix, sizeInBytes);
    cudaMalloc(&d_result, (sizeOfMatrix * sizeof(float)));

    start_s = clock();

    cudaMemcpy(d_matrix, matrix, sizeInBytes, cudaMemcpyHostToDevice);

    gelm<<<size / 1024 + 1, 1024/*, sizeOfMatrix + 1*/>>>(d_matrix, d_result, sizeOfMatrix, size);

    cudaMemcpy(result, d_result, (sizeOfMatrix * sizeof(float)), cudaMemcpyDeviceToHost);

    cout << "\nResult from GPU : (";

    for (int i = 0; i < sizeOfMatrix; i++)
    {
        cout << result[i];

        if(i < sizeOfMatrix - 1)
            cout << ", ";
    }

    cout << ")" << endl;

    stop_s = clock();

    float gpu = (stop_s-start_s)/float(CLOCKS_PER_SEC)*1000;
    cout << "time GPU: " << gpu << endl;

    cout << "Improvement: " << setprecision(4) << (cpu - gpu) / cpu << endl;
    cout << "Improvement: " << setprecision(4) << cpu / gpu << endl;

        // ****************** //

    cout << "\n";

    for(int i = 0; i < sizeOfMatrix; i++)
        delete[] coefMatrix[i];

    delete[] coefMatrix;

    delete[] constMatrix;

        // *********** //

    cudaFree(d_matrix);
    cudaFree(d_result);
    delete[] matrix;
    delete[] result;
    return 0;
}

