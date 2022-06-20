# Movement-Heat-Mapping

![image](https://user-images.githubusercontent.com/67343196/174458468-26823e10-f44f-4a30-b943-384fd3adf13d.png)

**You need to get weight and config files for this process.**

This project uses yolo model for detecting person of input. So you can choose which classes you want to detect or download which yolo pretrained model files you want for your process. 

You need to edit config and weight variables in your *\*py* file to load the correct model.

**Setup**

Clone repo and install *requirements.txt*.
<pre>
"pip install -r requirements.txt"
</pre>

Use <code>*heatmapvideo.py*</code> for heat mapping a video.
**Note:** The input video should be named <code>*videoinput.mp4*</code> and put in main directory.

Use <code>*heatmapimgset.py*</code> for heat mapping a set of images.
**Note:** The input image set should be put in <code>*imageset*</code> folder in main directory (you must create one if it doesn't exist).

Press <code>"E"</code> to stop the process.
