<div id="top"></div>

<br />
<div align="center">

<img src="https://civitai.com/favicon.ico" alt="Logo" width="80" height="80" style="border-radius: 25%;">

<a href="https://civitai.com">
  <h3 align="center">Civitai Downloader</h3>
</a>

  <p align="center">
    A script using Civitai's <a href="https://github.com/civitai/civitai/wiki/REST-API-Reference">REST-API-Reference</a> 
    <br />
    <br />
    <br />

  </p>
</div>

### Contents
<div id="index"></div>

* <p align="left"><a href="#prereq">Installation</a></p>
* <p align="left"><a href="#config">API Key</a></p>
* <p align="left"><a href="#usage">Usage</a></p>
* <p align="left"><a href="#credits">Credits</a></p>
* <p align="left"><a href="#todo">Todo's</a></p>
<p align="right">(<a href="#top">back to top</a>)</p>


### Pre-Requisites
<div id="prereq"></div>

This is an example of how to list things you need to use the software and how to install them.
* python 3.9 +

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/axsddlr/civitai_downloader.git
   ```
2. python
   ```sh
   python -m pip install -r requirements.txt
   ```
<p align="right">(<a href="#top">back to top</a>)</p>

## config.json
<div id="config"></div>

1. go to your Civitai account and go to `Settings` > `API`
2. create a name for your api key and click `Save`
3. copy the api key and paste it into `example.config.json` replacing `<api key here>`
4. rename `example.config.json` to `config.json`

```json
{
  "civitai_api_key": "<api key here>"
}
```

create a `id.txt` file and paste in links you want to download
![menu1.png](assets%2Fmenu1.png) ![menu2.png](assets%2Fmenu2.png)
![menu3.png](assets%2Fmenu3.png) ![menu4.png](assets%2Fmenu4.png)

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage
<div id="usage"></div>

   ```sh
  python main.py
   ```
extra options
   ```sh
  python main.py -h
   ```
<p align="right">(<a href="#top">back to top</a>)</p>

## Credits
<div id="credits"></div>

   ```markdown
  Clicky#6060 (initial creation idea)
   ```

## TODO
<div id="todo"></div>

   ```markdown
    1. auto download a preview image from item in API for each downloaded file
    2. allow user to specify if they want to download pickletensor or safetensors files
    3. allow user to specify if they want to download pruned version of the files
    4. sql database to store all downloaded files info
   ```

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>