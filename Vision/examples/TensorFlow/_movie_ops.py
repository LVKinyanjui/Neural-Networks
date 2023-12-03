#!/usr/bin/env python
# coding: utf-8

# # Class Implementation
# I find that this project is a little to complex to be merely procedural. I would like to encapsulate the logic above in a class. My reasons are as follows:
# 1. State Information
#     We need to keep track of the current state of our application. Is the window open, are we on the right site, have we started the download. By having class methods and object attributes, we can keep track.
# 2. Data. 
#     We have some information that is packaged with our program. It is best to encapsulate them as an object.

# In[2]:


import time
import subprocess
import pyautogui
from pygetwindow import PyGetWindowException
from pyscreeze import PyScreezeException


class MovieOps():
    
    def __init__(self, executable_path='C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe', page_url='https://fmovies.to/', process_name="brave.exe"):
        """
        Parameters
            executable_path: Full path to browser executable. The brave browser is recommended
            page_url: the movie webpage url
            process_name: the window or browser name. To be used to kill existing instances at initialization.
        """
        self.executable_path = executable_path
        self.page_url = page_url
        self.on_page = False
        
        # Kill existing window processes
        self.terminate(process_name)
        
        # Start a new instance
        subprocess.Popen(self.executable_path)
        
    def terminate(self, process_name="brave.exe"):
        # Use the taskkill command to terminate the process by name
        try:
            subprocess.run(["taskkill", "/F", "/IM", process_name], check=True)
            print(f"Process '{process_name}' terminated successfully.")
        except subprocess.CalledProcessError:
            print(f"Process '{process_name}' could not be terminated, or was already terminated.")
            
        
    def locate(self, image, confidence=0.9, region=(0, 0, 1366, 768)):
        loc = pyautogui.locateCenterOnScreen(image, confidence=confidence, region=region)
        if loc is not None:
            print(f"App Icon: {loc}")
            x, y = loc
            pyautogui.moveTo(x, y)
            return x, y
            # pyautogui.click()
    
    def open_window(self, title="New Tab - Brave"):
        try:
            # Try to locate image directly
            pyautogui.locateOnWindow(title=title, image="images/fmovies.jpg")
            
        except PyScreezeException:
            self.simple_switch(image="images/brave_icon.png")
            
            # # Partial fix: Close all brave instances and start a new one
            # self.terminate()
            # self.__init__()
            
        except PyGetWindowException:
            # Click brave
            self.simple_switch("images/brave_icon_open.jpg")

        # Address needle dimension(s) exceed the haystack image or region dimensions issue
        except ValueError:
            self.simple_switch("images/brave_icon.png")
            
    def simple_switch(self, image="images/brave_icon.png"):
        # Simple window switch
        ## Only if it is the only similarly named window
        active = pyautogui.getActiveWindow()
        if 'Brave' not in active.title:
            self.locate(image=image)
            pyautogui.click()
            
            # active_ = pyautogui.getActiveWindow()
            # if active_ is not None:
            #     active_.maximize()
            #     # pyautogui.screenshot("images/current_page.jpg")
            
    
    def load_fmovies(self, url):
        
        self.simple_switch()
        
        # Check if we are on a new tab
        # if yes proceed with search
        active = pyautogui.getActiveWindow()
        print(f"Active Tab: {active.title}")
        
        if 'New Tab' in active.title:
        
            # Clear any text in search bar
            pyautogui.keyDown('ctrl')
            pyautogui.press('a')
            pyautogui.keyUp('ctrl')
            pyautogui.press('backspace')

            # Send url and press enter
            pyautogui.typewrite(url)
            pyautogui.press('enter')
            
            
        elif 'FMovies' in active.title:
            self.on_page = True
            print("On Fmovies")
    
        # Send the keyboard shortcut Ctrl + 0 to set browser magnification to 100%
        pyautogui.hotkey('ctrl', '0')
        
    def search_movie(self, name, image="images/search_bar_fmovies_homepage.jpg", confidence=0.8):
        
        loc = self.locate(image=image, confidence=confidence)
        if loc is not None:
            pyautogui.click()
        
            pyautogui.moveRel(xOffset=-100, yOffset=0)
            pyautogui.click()

            pyautogui.keyDown('ctrl')
            pyautogui.press('a')
            pyautogui.keyUp('ctrl')
            pyautogui.typewrite(name)
            # Preliminary: Selecting movie
            # Select the first result
            pyautogui.moveRel(xOffset=0, yOffset=50, duration=1)
            pyautogui.click()
        else:
            print("Search Icon Not Found")


# ## Modifications
# 1. Have an `is_active` method to check whether brave is active window before progressing.

# In[ ]:




