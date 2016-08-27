#include <pxcdefs.h>
#include <pxcimage.h>
#include "FileOperation.hpp"
#include "Segmentation.hpp"

class Dataset: public FileOperation
{
public:
	Dataset(const string& Dir, int width, int height, float fps = 60);
	int dataAcquire();
	Mat PXCImage2Mat(PXCImage* pxc);

	inline string getSavePath(const string &dirPrefix, time_t slot, long framecnt);
	inline string getSaveFileName(time_t slot, long framecnt);
	inline string getSaveDirName(time_t slot);
	
	//
	int testSVM(string dir);
	int testRegion(string dir);
	//
	//demo with real-time capture video stream
	int show();
	//
	int savePCD(const string &outfilename, Segmentation &myseg);
	//
	inline int saveColor();
	inline int saveDepth();
	int saveRegions(vector<Rect> Boxes);
	//analysis from files
	//int Dataset::readSourceFromFile();
private:
	string dir_;
	string depthDir_;
	string colorDir_;
	string pcdDir_;

	PXCSizeI32 camera_;
	pxcF32 fps_;
	vector<PXCPoint3DF32> vertices_;
	//HOG_SVM classification_;
};


inline string Dataset::getSavePath(const string &dirPrefix, time_t slot, long framecnt)
{
	std::stringstream ss;
	ss << dirPrefix << "\\" << getSaveDirName(slot) << getSaveFileName(slot, framecnt);
	return ss.str();
}

inline string Dataset::getSaveFileName(time_t slot, long framecnt)
{
	return to_string(slot) + "-" + to_string(framecnt) +".png";
}

inline string Dataset::getSaveDirName(time_t slot)
{
	string res = "";
	tm time_slot;
	localtime_s(&time_slot, &slot);
	res += to_string(time_slot.tm_year + 1990);
	res += to_string(-(time_slot.tm_mon + 1));
	res += to_string(-(time_slot.tm_mday));
	res += to_string(-(time_slot.tm_hour));
	return res;
}