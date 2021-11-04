# Setting up the Environment

This document will allow you to setup and run the program on your local machine.

## Python Setup

This project uses python version `3.9.7`. The program is not guaranteed to work using another version of python.

## Setup your Virtual Environment

You will likely want to create a virtual environment to keep this project isolated. If you want all of the packages to be global instead, you may skip this step, although this is not recommended!

You can create a virtual environment directly from python using `venv`.

```
python3 -m venv /venv/yourVenvName
```

You want to create your venv in a directory called `venv/` or otherwise outside of this project to make sure the files contents of your environment are not picked up by Git.

Activate your virtual environment by running the `activate` script found in the `Scripts` folder of your venv for Windows systems, or the `bin` folder for POSIX.

```
.\venv\yourVenvName\Scripts\activate
```

## Package Installation

To install all the necessary libraries, perform a `pip install` from the `requirements.txt` file in the root directory.

```
pip install -r requirements.txt
```

### Linting

This will also install `pylint` for linting and `autopep8` for auto formatting. Be sure to select `pylint` as your linter if your IDE supports it, or simply run `pycodestyle` in the directory before comitting a change.

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

To setup your environment configuration, you must create a `.env` file in the root directory. To do this, copy and paste the `.env.example` file, which will contain all the current keys for the dotenv configuration.

Set the values in the `.env` file as needed.

You have additional configuration options via CLI arguments. Run the python script with `--help` or `-h` to see the CLI arguments including their descriptions.

For more information on configuration, please read [the Configuration documentation](./Configuration.md).
