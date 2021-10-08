# Setting up the Environment

## Installing Packages

To install the required packages:

```
pip install -r requirements.txt
```

## Downloading FFmpeg

**FFmpeg** is a tool that is used as a backend to librosa's reading functionality, which allows librosa to read `.mp3` files.

The following sections outlines methods on integrating the tool with librosa.

### No environment

If you are not using a virtual environment, you should be able to do

```
pip install ffmpeg
```

and have it work from there. If this doesn't work, then adding the `ffmpeg.exe` path to your PATH environment variables should make it work.

### With a Conda Environment

With Anaconda, you should be able to use the `conda-forge` command to setup ffmpeg with librosa.

```
conda install -c conda-forge ffmpeg librosa
```

### With a Virtual Environment (or if other methods failed!)

The simplest method would be to add the `ffmpeg.exe` file to the root of the project. You can find [.exe file downloads here](https://github.com/BtbN/FFmpeg-Builds/releases).

## Configuration

### .env

Duplicate the `.env.example` file into a `.env` file.

### CLI Arguments

Run the python script with the `--help` or `-h` flags to see the CLI arguments
