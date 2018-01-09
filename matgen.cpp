#include <iostream>
#include <fstream>

using namespace std;

int main()
{
	ofstream outFile("inputMatrix.txt");
	cout << "Enter size: " << endl;
	int size;

	cin >> size;

	outFile << size << endl;

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size + 1; j++)
		{
			if(j >= i)
				outFile << 1 << " ";
			else
				outFile << 0 << " ";
		}
		outFile << endl;
	}

	outFile.close();
}
