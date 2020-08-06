package com.hfad.yoloandroid;

import android.graphics.Bitmap;
import android.graphics.Camera;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
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
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.core.CvType.CV_8UC4;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;

    Mat lascaFrame, frame, avfilter, framemean, frame2mean, prevLascaFrame, averageLascaFrame, norm_m_calc1, norm_m_calc2, norm_m, dyn_calc1, dyn_calc2, dyn_contrast;
    Camera mCamera;
    List<Mat> frameAccumulator;
    //window size for moving averages
    int wsize = 5;
    int framestoaverage = 30;
    //count integer for accumulator
    int count = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.setMinimumHeight(3024);
        cameraBridgeViewBase.setMinimumWidth(4032);
        cameraBridgeViewBase.setMaxFrameSize(4032, 3024);
        cameraBridgeViewBase.setCameraIndex(-1);

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

    private void getLASCA(Mat inputMat) {
        //The input here is just inputFrame.rgba(); from the onCameraFrame method
        prevLascaFrame = lascaFrame;
        Core.extractChannel(frame, frame, 0);
        Core.normalize(frame, frame, 0,1, Core.NORM_MINMAX, CvType.CV_64F);
        //local blur
        Imgproc.blur(frame, framemean, new Size(wsize,wsize), new Point(-1,-1));
        Imgproc.GaussianBlur(frame, frame2mean, new Size(wsize,wsize), 0);
        Core.divide(frame2mean, framemean, lascaFrame);
        Core.addWeighted(lascaFrame, 0.5, prevLascaFrame, 0.5, 0, lascaFrame, CV_64F);
        Core.normalize(lascaFrame, lascaFrame, 0, 255, Core.NORM_MINMAX, CV_8U);
        Imgproc.applyColorMap(lascaFrame, lascaFrame, 2);
    }

    public void getLASCAAverage(Mat inputmat){
        Core.extractChannel(frame, frame, 0);
        Core.normalize(frame, frame, 0,1, Core.NORM_MINMAX, CvType.CV_64F);

        Imgproc.blur(frame, framemean, new Size(wsize,wsize), new Point(-1,-1));
        Core.multiply(frame, frame, frame2mean);
        Imgproc.blur(frame, frame2mean, new Size(wsize,wsize), new Point(-1,-1));
        Core.sqrt(framemean, framemean);
        //Core.subtract(framemean, new Scalar(1), framemean);
        Core.divide(frame2mean, framemean, lascaFrame);
        Core.sqrt(lascaFrame, lascaFrame);
        //we not have our lasca frame so we can average them over time

        frameAccumulator.set(count, lascaFrame);
        count = count + 1;
        if (count == framestoaverage){
            count = 0;
            averageLascaFrame.setTo(new Scalar(0));
            //Core.add(averageLascaFrame, new Scalar(0.01), averageLascaFrame);
        }

        for (int i = 0; i < framestoaverage; i++) {
            //Imgproc.accumulateWeighted(frameAccumulator.get(i), averageLascaFrame, 0.1);
            Imgproc.accumulate(frameAccumulator.get(i), averageLascaFrame);
        }
        Core.normalize(averageLascaFrame, frame, 0, 255, Core.NORM_MINMAX, CV_8U);
        Imgproc.applyColorMap(frame, frame, 2);
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //gets triggered 20 or 30 times a second. This is where we do any maths on our camera

        frame = inputFrame.rgba();
        Core.extractChannel(frame, frame, 0);
        frame.convertTo(frame, CV_64F);
        frameAccumulator.set(count, frame);

        for (int i = 0; i < framestoaverage - 1; i++) {
            //Imgproc.accumulateWeighted(frameAccumulator.get(i), averageLascaFrame, 0.1);
            Imgproc.accumulate(frameAccumulator.get(i), norm_m_calc1);
        }
        Core.divide(norm_m_calc1, new Scalar(framestoaverage), norm_m_calc1);
        for (int i = 1; i < framestoaverage; i++) {
            //Imgproc.accumulateWeighted(frameAccumulator.get(i), averageLascaFrame, 0.1);
            Imgproc.accumulate(frameAccumulator.get(i), norm_m_calc2);
        }
        Core.divide(norm_m_calc2, new Scalar(framestoaverage), norm_m_calc2);
        dyn_calc1 = norm_m_calc1.clone();
        dyn_calc2 = norm_m_calc2.clone();
        Imgproc.blur(norm_m_calc1, norm_m_calc1, new Size(wsize,wsize), new Point(-1,-1));
        Imgproc.blur(norm_m_calc2, norm_m_calc2, new Size(wsize,wsize), new Point(-1,-1));
        Core.multiply(norm_m_calc1, norm_m_calc2, norm_m);

        //calculate dynamic part
        Core.subtract(dyn_calc2, dyn_calc1, dyn_calc1);
        Core.multiply(dyn_calc1, dyn_calc1, dyn_calc1);
        Core.divide(dyn_calc1, new Scalar(2), dyn_calc1);
        Core.divide(dyn_calc1, new Scalar(0.4252), dyn_calc1);
        Imgproc.threshold(dyn_calc1, dyn_calc1, 0.001, 1, Imgproc.THRESH_TOZERO);
        Core.normalize(dyn_calc1, frame, 0, 255, Core.NORM_MINMAX, CV_8U);
        Imgproc.applyColorMap(frame, frame, 2);


        //Imgproc.blur(frame, framemean, new Size(wsize,wsize), new Point(-1,-1));
        //Core.multiply(frame, frame, frame2mean);
        //Imgproc.blur(frame, frame2mean, new Size(wsize,wsize), new Point(-1,-1));
        //Core.sqrt(framemean, framemean);
        //Core.subtract(framemean, new Scalar(1), framemean);
        //Core.divide(frame2mean, framemean, lascaFrame);
        //Core.sqrt(lascaFrame, lascaFrame);
        //we not have our lasca frame so we can average them over time

        return inputFrame.rgba();
        //return frame;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        int rowsize = 1080;
        int colsize = 1920;
        lascaFrame = new Mat(rowsize, colsize, CV_64F);
        frame = new Mat(rowsize, colsize, CV_64F);
        avfilter = new Mat(rowsize, colsize, CV_64F);
        framemean = new Mat(rowsize, colsize, CV_64F);
        frame2mean = new Mat(rowsize, colsize, CV_64F);
        averageLascaFrame = new Mat(rowsize, colsize, CV_64F);
        norm_m = new Mat(rowsize, colsize, CV_64F);
        norm_m_calc1 = new Mat(rowsize, colsize, CV_64F);
        norm_m_calc2 = new Mat(rowsize, colsize, CV_64F);
        dyn_calc1 = new Mat(rowsize, colsize, CV_64F);
        dyn_calc2 = new Mat(rowsize, colsize, CV_64F);
        dyn_contrast = new Mat(rowsize, colsize, CV_64F);

        frameAccumulator = new ArrayList<Mat>();
        for (int i = 0; i < framestoaverage; i ++) {
        frameAccumulator.add(frame);
        }





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
