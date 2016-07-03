#include "Dataset.hpp"
#include "DrawWorld.hpp"
#include "Macro.hpp"
#include "Opencv.hpp"
#include "Realsense.hpp"

// Locate windows position
void placeWindows(int topk)
{
	int interval = 300;
	cv::namedWindow("depth");
	cv::namedWindow("color");
	cv::namedWindow("before merging");
	cv::namedWindow("segmentation");
	cv::namedWindow("classification");
	cv::namedWindow("regions");
	cv::moveWindow("depth", 0, 0);
	cv::moveWindow("color", interval, 0);
	cv::moveWindow("segmentation", 3 * interval, 0);
	cv::moveWindow("before merging", 2 * interval, 0);
	cv::moveWindow("classification", 0, 2 * interval);
	cv::moveWindow("regions", interval, 2 * interval);
	for (int k = 0; k < topk; k++) {
		cv::namedWindow(to_string(k));
		cv::moveWindow(to_string(k), k * interval, interval);
	}
}

//Dir example: "..\\saveData\\"
Dataset::Dataset(const string& Dir, int width, int height, float fps /*= 60*/) :
dir_(Dir), depthDir_(Dir + "\\depth\\"), rgbDir_(Dir + "\\rgb\\"), pcdDir_(Dir + "\\pcd\\"), fps_(fps)
{
	camera_.width = width;
	camera_.height = height;
}

//Convert RealSense's PXCImage to Opencv's Mat
Mat Dataset::PXCImage2Mat(PXCImage* pxc)
{
	if (!pxc)	return Mat(0, 0, 0);
	PXCImage::ImageInfo info = pxc->QueryInfo();
	PXCImage::ImageData data;
	Mat cvt;
	if (info.format & PXCImage::PIXEL_FORMAT_YUY2) {	//颜色数据
		if (pxc->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &data) < PXC_STATUS_NO_ERROR)
			return  Mat(0, 0, 0);
		cvt = Mat(info.height, info.width, CV_8UC3, (void*)data.planes[0],data.pitches[0]/sizeof(uchar));
	}
	else if (info.format & PXCImage::PIXEL_FORMAT_DEPTH) {//深度数据
		if (pxc->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_DEPTH, &data) < PXC_STATUS_NO_ERROR)
			return  Mat(0, 0, 0);
		cvt = Mat(info.height, info.width, CV_16U, (void*)data.planes[0], data.pitches[0] / sizeof(uchar));	//Mat初始化是按照长宽来定的
	}
	pxc->ReleaseAccess(&data);
	return cvt;
}

//Save PXC Point Cloud to PCD file
int Dataset::savePCD(const string& outfilename, Segmentation &myseg)
{
	ofstream ofs(outfilename);
	ofs << "# .PCD v0.7 - Point Cloud Data file format" << endl;
	ofs << "VERSION 0.7" << endl;
	ofs << "FIELDS x y z" << endl;
	ofs << "SIZE 4 4 4" << endl;
	ofs << "TYPE F F F" << endl;
	ofs << "COUNT 1 1 1" << endl;
	ofs << "WIDTH " << myseg.mainRegions_[0].size() << endl;
	ofs << "HEIGHT 1" << endl;
	ofs << "VIEWPOINT 0 0 0 1 0 0 0" << endl;
	ofs << "POINTS " << myseg.mainRegions_[0].size() << endl;
	ofs << "DATA ascii" << endl;
	//double scale = 1. / 300;
	double scale = 1. / 330;
	//vector<PXCPoint3DF32> obj_cloud;
	for (auto& p : myseg.mainRegions_[0]) {
		p += p;
		PXCPoint3DF32 ppp = vertices_[p.y * 640 + p.x];
		ofs << ppp.x*scale << " " << ppp.y*scale << " " << ppp.z*scale << endl;
		//obj_cloud.push_back(vertices[p.y * 640 + p.x]);
	}
	ofs.close();
	return 0;
}

//Acquire color/depth/pcd data
int Dataset::dataAcquire()
{
	//Define variable
	Mat color, depth, display;
	vector<PXCPoint3DF32> vertices(camera_.height*camera_.width);
	PXCSession *pxcsession;
	PXCSenseManager *pxcsm;
	PXCCapture::Device *pxcdev;
	PXCProjection *projection;
	PXCCapture::Sample *sample;
	PXCImage *pxcdepth,*pxccolor;
	long framecnt;
	//Configure RealSense
	pxcsession = PXCSession::CreateInstance();
	pxcsm = pxcsession->CreateSenseManager();
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, camera_.width, camera_.height, fps_);
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, camera_.width, camera_.height, fps_);
	pxcsm->Init();
	pxcdev = pxcsm->QueryCaptureManager()->QueryDevice();
	pxcdev->SetDepthConfidenceThreshold(4);
	if (!pxcdev) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK SenseManager");
		return -1;
	}
	projection = pxcdev->CreateProjection();
	if (!projection) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK Projection");
		return -1;
	}
	//Configure Point Cloud Show
	DrawWorld dw(pxcsession, camera_);
	PXCPoint3DF32 light = { .5, .5, 1.0 };
	//Configure Segmentation
	unsigned topk = 5;
	short threshold = 3;
	Segmentation myseg(320, 240, topk, threshold);
	//Detect each video frame
	for (framecnt = 1; -1 == waitKey(1); ++framecnt) {
		if (pxcsm->AcquireFrame(true) < PXC_STATUS_NO_ERROR)	break;
		//Query the realsense color and depth, and project depth to color
		try{
			sample = pxcsm->QuerySample();
			pxcdepth = sample->depth;
			pxccolor = sample->color;
			pxcdepth = projection->CreateDepthImageMappedToColor(pxcdepth, pxccolor);
			//Generate and Show 3D Point Cloud
			pxcStatus sts = projection->QueryVertices(pxcdepth, &vertices[0]);
			if (sts >= PXC_STATUS_NO_ERROR) {
				PXCImage* drawVertices = dw.DepthToWorldByQueryVertices(vertices, pxcdepth, light);
				if (drawVertices){
					Mat display = PXCImage2Mat(drawVertices);
					imshow("display", display);
				}
			}
			//waitKey(-1);
			

			////project to world
			//projection->QueryVertices(pxcdepth, &vertices[0]);
			//
			//ofstream ofs("123.txt");
			//for (auto v : vertices) {
			//	if (v.x || v.y || v.z)
			//		ofs << "v " << v.x << " " << v.y << " " << v.z << endl;
			//}
			//cout << endl;



			//pxcdepth = projection->CreateColorImageMappedToDepth(pxcdepth,pxccolor);
			depth = PXCImage2Mat(pxcdepth);
			color = PXCImage2Mat(pxccolor);
			if (!depth.cols || !color.cols)	continue;

			Mat depth2, color2;
			resize(depth, depth2, Size(320, 240));
			resize(color, color2, Size(320, 240));

			imshow("depth", 65535 / 1200 * depth2);
			imshow("color", color2);

			//myseg.completeDepth(depth);
			if (framecnt & 1) {
				myseg.Segment(depth2, color2);
				myseg.clear();
			}


			//Segment by Depth
			//bfs(depth,vector<vector<Point>>)

			////Construct the filename to save
			//time_t slot = time(0);
			//std::string cpath = getSavePath(rgbDir_, slot, framecnt);
			//std::string dpath = getSavePath(depthDir_, slot, framecnt);

			////Save image
			//imwrite(cpath, color);
			//imwrite(dpath, depth);

			//double dmax, dmin;
			//minMaxLoc(depth, &dmin, &dmax);
			//MESSAGE_COUT("MIN-MAX", dmin << " , " << dmax);

			//Show image

			//resize(color, color, cv::Size(640, 480));
			//resize(depth, depth, cv::Size(640, 480));

			
			//imshow("Depth", 256 * 255 / 1200 * depth);
			//imshow("COLOR", color);
			
			
			//Release Realsense SDK memory and read next frame 
			pxcdepth->Release();
			pxcsm->ReleaseFrame();
		}
		catch (cv::Exception e){
			MESSAGE_COUT("ERROR", e.what());
		}
	}
	return 1;

}


int Dataset::show()
{
	/************************************************************************/
	/* Define variable                                                      */
	/************************************************************************/
	Mat color, depth, display;
	PXCSession *pxcsession;
	PXCSenseManager *pxcsm;
	PXCCapture::Device *pxcdev;
	PXCProjection *projection;
	PXCCapture::Sample *sample;
	PXCImage *pxcdepth, *pxccolor;
	long framecnt;
	/************************************************************************/
	/* Configure RealSense                                                  */
	/************************************************************************/
	pxcsession = PXCSession::CreateInstance();
	pxcsm = pxcsession->CreateSenseManager();
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, camera_.width, camera_.height, fps_);
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, camera_.width, camera_.height, fps_);
	//Configure Draw
	PXCPoint3DF32 light = { .5, .5, 1.0 };
	DrawWorld dw(pxcsession, camera_);
	//Configure Segmentation
	unsigned topk = 5;
	short threshold = 3;
	Segmentation myseg(320, 240, topk, threshold);
	//placeWindows(topk);
	
	
	//Query Information
	pxcsm->Init();
	pxcdev = pxcsm->QueryCaptureManager()->QueryDevice();

	//PXCPointF32 cf = pxcdev->QueryColorFocalLength();
	//PXCPointF32 df = pxcdev->QueryDepthFocalLength();
	//pxcF32 cfmm = pxcdev->QueryColorFocalLengthMM();
	//pxcF32 dfmm = pxcdev->QueryDepthFocalLengthMM();
	//PXCPointF32 dpp = pxcdev->QueryDepthPrincipalPoint();
	//PXCPointF32 cpp = pxcdev->QueryColorPrincipalPoint();
	//PXCPointF32 dview = pxcdev->QueryDepthFieldOfView();
	//PXCRangeF32 drange = pxcdev->QueryDepthSensorRange();
	//PXCCapture::DeviceInfo ppp;
	//pxcdev->QueryDeviceInfo(&ppp);
	//cout << cfmm << endl;
	//cout << dfmm << endl;
	//cout << endl;


	//setthreshold
	//pxcdev->SetDepthConfidenceThreshold(4);


	if (!pxcdev) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK SenseManager");
		return -1;
	}
	projection = pxcdev->CreateProjection();
	if (!projection) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK Projection");
		return -1;
	}

	//calibration
	//PXCCalibration *calib = projection->QueryInstance<PXCCalibration>();
	//PXCCalibration::StreamCalibration sc;
	//PXCCalibration::StreamTransform st;
	//calib->QueryStreamProjectionParametersEx(PXCCapture::StreamType::STREAM_TYPE_DEPTH,
	//	PXCCapture::Device::StreamOption::STREAM_OPTION_DEPTH_PRECALCULATE_UVMAP,
	//	&sc, &st);
	//cout << endl;

	/************************************************************************/
	/* Detect each video frame                                              */
	/************************************************************************/
	vertices_.resize(camera_.height*camera_.width);
	time_t now;
	for (	framecnt = 1, now = time(0);
			-1 == waitKey(1); 
			++framecnt, now = time(0), pxcdepth->Release(), pxcsm->ReleaseFrame()	) 
	{
		if (pxcsm->AcquireFrame(true) < PXC_STATUS_NO_ERROR)	break;
		//Query the realsense color and depth, and project depth to color
		sample = pxcsm->QuerySample();
		pxcdepth = sample->depth;
		pxccolor = sample->color;
		pxcdepth = projection->CreateDepthImageMappedToColor(pxcdepth, pxccolor);

		pxcStatus sts = projection->QueryVertices(pxcdepth, &vertices_[0]);
		if (sts >= PXC_STATUS_NO_ERROR) {
			PXCImage* drawVertices = dw.DepthToWorldByQueryVertices(vertices_, pxcdepth, light);
			if (drawVertices){
				display = PXCImage2Mat(drawVertices);
				//imshow("display", display);
			}
		}

		depth = PXCImage2Mat(pxcdepth);
		color = PXCImage2Mat(pxccolor);
		if (!depth.cols || !color.cols)	continue;

		Mat depth2, color2;
		resize(depth, depth2, Size(320, 240));
		resize(color, color2, Size(320, 240));

		Mat show = color2.clone();

		//myseg.completeDepth(depth);

		imshow("depth", 65535 / 1200 * depth2);
		//imshow("color", color2);

		myseg.Segment(depth2, color2);
		resize(display, display, Size(320, 240));


		int S = 1;
		//myseg.mainRegions_.size();
		for (int k = 0; k < S; k++){
			for (auto p : myseg.mainRegions_[k]) {
				show.at<Vec3b>(p) = display.at<Vec3b>(p);
			}
			imshow(to_string(k), show);
		}


		//vector<PXCPoint3DF32> obj_cloud;
		//static Point dir[5] = { { 0, 0 }, { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
		//static Rect range(0, 0, 640, 480);
		//for (int k = 0; k < S; k++){
		//	Mat show = Mat::zeros(Size(640, 480), CV_8U);

		//	for (auto p : myseg.mainRegions_[k]) {
		//		cout << p.x <<" "<< p.y << endl;
		//		for (auto d : dir) {
		//			Point pp = p * 2 + d;
		//			if (!pp.inside(range)) continue;
		//			cout << p.x << " " << p.y << endl;
		//			obj_cloud.push_back(vertices[pp.y * 640 + pp.x]);
		//			show.at<Vec3b>(pp) = display.at<Vec3b>(pp);
		//		}
		//	}
		//	imshow(to_string(k), show);
		//}


		//myseg.Segment(depth, color);

		

		if (waitKey(1) == ' ') {
			//string filename = getSaveFileName(now, framecnt);
			//string pcdFilepath = pcdDir_ + filename + ".pcd";
			//string rgbFilepath = rgbDir_ + filename;
			//string depthFilepath = depthDir_ + filename;
			//{//record regions;
			//	string recordPath = dir_ + filename + ".txt";
			//	ofstream record(recordPath);
			//	for (auto &seg : myseg.mainRegions_) {
			//	}
			//	record.close();
			//}
			//string filename = "realsense.pcd";
			string filename = getSaveFileName(now, framecnt) + ".pcd";
			cout << filename << endl;
			
			savePCD(filename, myseg);

			cout << "OK" << endl;


 			waitKey(-1);
		}
		//Release
		myseg.clear();
	}
	return 1;
}



//int Dataset::show()
//{
//	//Define variable
//	Mat color, depth, display;
//	vector<PXCPoint3DF32> vertices(camera_.height*camera_.width);
//	PXCSession *pxcsession;
//	PXCSenseManager *pxcsm;
//	PXCCapture::Device *pxcdev;
//	PXCProjection *projection;
//	PXCCapture::Sample *sample;
//	PXCImage *pxcdepth, *pxccolor;
//	long framecnt;
//	//Configure RealSense
//	pxcsession = PXCSession::CreateInstance();
//	pxcsm = pxcsession->CreateSenseManager();
//	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, camera_.width, camera_.height, fps_);
//	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, camera_.width, camera_.height, fps_);
//	//Configure Draw
//	PXCPoint3DF32 light = { .5, .5, 1.0 };
//	DrawWorld dw(pxcsession, camera_);
//	//Configure Segmentation
//	unsigned topk = 5;
//	short threshold = 3;
//	Segmentation myseg(320, 240, topk, threshold);
//	placeWindows(topk);
//	//Query Information
//	pxcsm->Init();
//	pxcdev = pxcsm->QueryCaptureManager()->QueryDevice();
//
//	PXCPointF32 cf = pxcdev->QueryColorFocalLength();
//	PXCPointF32 df = pxcdev->QueryDepthFocalLength();
//	pxcF32 cfmm = pxcdev->QueryColorFocalLengthMM();
//	pxcF32 dfmm = pxcdev->QueryDepthFocalLengthMM();
//	PXCPointF32 dpp = pxcdev->QueryDepthPrincipalPoint();
//	PXCPointF32 cpp = pxcdev->QueryColorPrincipalPoint();
//	PXCPointF32 dview = pxcdev->QueryDepthFieldOfView();
//	PXCRangeF32 drange = pxcdev->QueryDepthSensorRange();
//	PXCCapture::DeviceInfo ppp;
//	pxcdev->QueryDeviceInfo(&ppp);
//	cout << cfmm << endl;
//	cout << dfmm << endl;
//	cout << endl;
//
//	
//	//setthreshold
//	//pxcdev->SetDepthConfidenceThreshold(4);
//
//
//	if (!pxcdev) {
//		MESSAGE_COUT("ERROR", "Failed to create an SDK SenseManager");
//		return -1;
//	}
//	projection = pxcdev->CreateProjection();
//	if (!projection) {
//		MESSAGE_COUT("ERROR", "Failed to create an SDK Projection");
//		return -1;
//	}
//
//	//calibration
//	//PXCCalibration *calib = projection->QueryInstance<PXCCalibration>();
//	//PXCCalibration::StreamCalibration sc;
//	//PXCCalibration::StreamTransform st;
//	//calib->QueryStreamProjectionParametersEx(PXCCapture::StreamType::STREAM_TYPE_DEPTH,
//	//	PXCCapture::Device::StreamOption::STREAM_OPTION_DEPTH_PRECALCULATE_UVMAP,
//	//	&sc, &st);
//	//cout << endl;
//
//	//Detect each video frame
//	for (framecnt = 1; -1 == waitKey(1); framecnt++) {
//		if (pxcsm->AcquireFrame(true) < PXC_STATUS_NO_ERROR)	break;
//		//Query the realsense color and depth, and project depth to color
//		sample = pxcsm->QuerySample();
//		pxcdepth = sample->depth;
//		pxccolor = sample->color;
//		pxcdepth = projection->CreateDepthImageMappedToColor(pxcdepth, pxccolor);
//
//		pxcStatus sts = projection->QueryVertices(pxcdepth, &vertices[0]);
//		if (sts >= PXC_STATUS_NO_ERROR) {
//			PXCImage* drawVertices = dw.DepthToWorldByQueryVertices(vertices, pxcdepth, light);
//			if (drawVertices){
//				display = PXCImage2Mat(drawVertices);
//				//imshow("display", display);
//			}
//		}
//
//		depth = PXCImage2Mat(pxcdepth);
//		color = PXCImage2Mat(pxccolor);
//		if (!depth.cols || !color.cols)	continue;
//
//		Mat depth2, color2;
//		resize(depth, depth2, Size(320, 240));
//		resize(color, color2, Size(320, 240));
//
//		myseg.completeDepth(depth);
//
//		imshow("depth", 65535 / 1200 * depth2);
//		imshow("color", color2);
//		Mat show = color2.clone();
//			
//		myseg.Segment(depth2, color2);
//		resize(display, display, Size(320, 240));
//			
//		
//		int S = 1;
//		myseg.mainRegions_.size();
//
//
//		//set<Point> obj_point ;
//		vector<PXCPoint3DF32> obj_cloud;
//		static Point dir[5] = { { 0, 0 }, { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
//		static Rect range(0, 0, 640, 480);
//		for (int k = 0; k < S; k++){
//			for (auto p : myseg.mainRegions_[k]) {
//				show.at<Vec3b>(p) = display.at<Vec3b>(p);
//				//for (auto d : dir) {
//				//	Point pp = p * 2 + d;
//				//	if (!pp.inside(range)) continue;
//				//	obj_point.insert(pp);
//				//}
//			}
//			imshow(to_string(k), show);
//		}
//
//
//		//vector<PXCPoint3DF32> obj_cloud;
//		//static Point dir[5] = { { 0, 0 }, { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
//		//static Rect range(0, 0, 640, 480);
//		
//		//for (int k = 0; k < S; k++){
//		//	Mat show = Mat::zeros(Size(640, 480), CV_8U);
//		//	
//		//	for (auto p : myseg.mainRegions_[k]) {
//		//		//cout << p.x <<" "<< p.y << endl;
//		//		for (auto d : dir) {
//		//			Point pp = p*2 + d;
//		//			if (!pp.inside(range)) continue;
//		//			//cout << p.x << " " << p.y << endl;
//		//			obj_cloud.push_back(vertices[pp.y * 640 + pp.x]);
//		//			//show.at<Vec3b>(pp) = display.at<Vec3b>(pp);
//		//		}
//		//	}			
//		//	imshow(to_string(k), show);
//		//}
//
//
//		myseg.Segment(depth, color);
//
//		myseg.clear();
//			
//		if (waitKey(1) == ' ') {
//			ofstream ofs("real-cloud.obj");
//			for (auto o : obj_cloud) {
//				ofs << "v " << o.x << " " << o.y << " " << o.x << endl;
//			}
//			waitKey(-1);
//		}
//
//		//Release
//		pxcdepth->Release();
//		pxcsm->ReleaseFrame();
//	}
//	return 1;
//}

