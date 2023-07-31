install dependencies
```
sudo apt update && sudo apt install ffmpeg python3-venv python3-tk nvidia-cudnn -y
```

start env
```
python3 -m venv env
```

enter venv
```
. env/bin/activate
```

```
pip install -r requirements.txt
```



if anything goes wrong please check docs on:
https://github.com/guillaumekln/faster-whisper