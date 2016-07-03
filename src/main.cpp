#include "Dataset.hpp"
int main(int argc, char** argv)
{
	string save_dir_path = "C:\\Users\\IDLER\\Documents\Visual Studio 2013\\Projects\\Realsense\\DataAcquire\\savedData";
	////F200
	Dataset data(save_dir_path, 640, 480, 30);
	data.dataAcquire();
	return 1;
}
