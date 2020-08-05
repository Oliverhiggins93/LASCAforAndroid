package com.hfad.yoloandroid;

import android.graphics.Bitmap;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.CvType.CV_16U;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_8U;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    Mat lascaFrame, frame, avfilter, framemean, frame2mean;
    List<Mat> allChannels;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);


        //next lines of code check that everything loads correctly with a switch case
        //if loaded correctly we enable the view
        baseLoaderCallback = new BaseLoaderCallback(this){
            @Override
            public void onManagerConnected(int status){
                super.onManagerConnected(status);
                switch(status){
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };
    }

    private Mat getLASCA(Mat singleChannel) {

        Imgproc.blur(singleChannel, avfilter, new Size(5,5), new Point(-1,-1));


        return avfilter;


    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //gets triggered 20 or 30 times a second
        //this is where we do any maths on our camera
        //java has no numpy so we use matrix objects
        //going to calculate the lasca here as it seems to prevent memory leaks
        frame = inputFrame.rgba();

        //extract red channel
        Core.extractChannel(frame, frame, 0);
        Core.normalize(frame, frame, 0,1, Core.NORM_MINMAX, CvType.CV_64F);
        //local blur
        int wsize = 21;
        Imgproc.blur(frame, framemean, new Size(wsize,wsize), new Point(-1,-1));
        Imgproc.GaussianBlur(frame, frame2mean, new Size(wsize,wsize), 0);
        Core.divide(frame2mean, framemean, lascaFrame);

        Core.normalize(lascaFrame, lascaFrame, 0, 255, Core.NORM_MINMAX, CV_8U);
        Imgproc.applyColorMap(lascaFrame, lascaFrame, 2);

        //local blur for squared version
        //Core.multiply(frame, frame, frame2mean);
        //Imgproc.blur(frame2mean, frame2mean, new Size(5,5), new Point(-1,-1));
        //calculate local contrast
        //Core.multiply(framemean, framemean, framemean);
        //Core.subtract(framemean, new Scalar(1), framemean);
        //Core.divide(frame2mean, framemean, lascaFrame);
        //Core.normalize(lascaFrame, lascaFrame, 0, 255, Core.NORM_MINMAX, CV_8U);
        //Imgproc.applyColorMap(lascaFrame, lascaFrame, 2);



        //Mat tmp = new Mat (1080, 1920, CvType.CV_8U, new Scalar(4));
        //Bitmap bmp = Bitmap.createBitmap(1920, 1080, Bitmap.Config.ARGB_8888);

        //Utils.matToBitmap(tmp, bmp);

        return lascaFrame;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        allChannels = new ArrayList<>(3);
        lascaFrame = new Mat(1080, 1920, CV_64F);
        frame = new Mat(1080, 1920, CV_64F);
        avfilter = new Mat(1080, 1920, CV_64F);
        framemean = new Mat(1080, 1920, CV_64F);
        frame2mean = new Mat(1080, 1920, CV_64F);



    }

    @Override
    public void onCameraViewStopped() {
        //we can use this to save an image or something when the camera view is stopped

    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(), "There is a problem.", Toast.LENGTH_SHORT).show();
        }
        else {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}
