#include "Dataset.hpp"
int main(int argc, char** argv)
{
	string save_dir_path = ".\\savedData\\";
	Dataset data(save_dir_path, 640, 480, 30);
	data.dataAcquire();

	return 1;
}
