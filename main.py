import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import CGAN

import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
# to use buttons:
from kivy.uix.button import Button
from kivy.uix.switch import Switch
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
import sys
from kivy.core.window import Window
from kivy.config import Config
from kivy.uix.slider import Slider
from  kivy.uix.filechooser import FileChooserListView

from kivy.utils import get_color_from_hex

kivy.require("1.10.1")
Config.set('graphics', 'width', '640')
Config.set('graphics', 'height', '800')

model_file_path = ''

alt_path = None
is_training = False
default_font_size = 50

class FirstPage(GridLayout):
    # runs on initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.next_button = Button(size=(640, 300), pos=(0, 400), text="Next", font_size=default_font_size)
        self.next_button.bind(on_press=self.next)
        self.add_widget(self.next_button)

        self.choose_file_button = Button(size=(640, 300), pos=(0, 50), text="Choose weight file", font_size=default_font_size)
        self.choose_file_button.bind(on_press=self.choose_file)
        self.add_widget(self.choose_file_button)


    def next(self, instance):
        chat_app.screen_manager.current = 'MainMenu'


    def choose_file(self, instance):
        global alt_path
        from plyer import filechooser
        path = filechooser.open_file(title="Pick a ckpt file..",
                                     filters=[("weight files", "*.ckpt*")])[0]
        alt_path = path[:path.find("ckpt") + 4].replace('//', '/')
        print(alt_path)


class MainPage(GridLayout):
    # runs on initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.cols = 1  # used for our grid

        self.random_face_button = Button(size=(640, 240), pos=(0, 510), text="Generate Random Face", font_size=default_font_size)
        self.random_face_button.bind(on_press=self.random_face)
        # self.add_widget(Label())  # just take up the spot.
        self.add_widget(self.random_face_button)

        self.sketch_artist_button = Button(size=(640, 240), pos=(0, 260), text="Sketch artist", font_size=default_font_size)
        self.sketch_artist_button.bind(on_press=self.sketch_artist)
        # self.add_widget(Label())  # just take up the spot.
        self.add_widget(self.sketch_artist_button)

        # add our button.
        self.face_manipulation_button = Button(size=(640, 240), pos=(0, 10), text="Face Manipulations", font_size=default_font_size)
        self.face_manipulation_button.bind(on_press=self.face_manipulation)
        # self.add_widget(Label())  # just take up the spot.
        self.add_widget(self.face_manipulation_button)

    def random_face(self, instance):
        chat_app.screen_manager.current = 'RandomFace'

    def sketch_artist(self, instance):
        chat_app.screen_manager.current = 'SketchArtist'

    def face_manipulation(self, instance):
        chat_app.screen_manager.current = 'FaceManipulation'


class RamdomFacePage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Window.size = (640, 700)
        self.cross_img = Image(source="", size=(640, 640), pos=(0, 120))

        self.add_widget(self.cross_img)

        self.noise_slider = Slider(min=-1, max=1, value=0, pos=(290, 40))
        self.add_widget(self.noise_slider)

        self.Generate_button = Button(text="Generate", size=(320, 60), pos=(0, 0), font_size=default_font_size)
        self.Generate_button.bind(on_press=self.Generate_image)
        self.add_widget(self.Generate_button)

        self.back_button = Button(text="Back", size=(320, 60), pos=(320, 0), font_size=default_font_size)
        self.back_button.bind(on_press=self.back)
        self.add_widget(self.back_button)



        self.switch = Switch(size=(50, 60), pos=(30, 50))
        self.switch.bind(active=self.switch_training)
        self.add_widget(self.switch)

    def Generate_image(self, instance):
        if cgan.test(is_training=is_training, desc='', noise=self.noise_slider.value, alt_path=alt_path):
            # self.cross_img = Image(source=cgan.test_dir + cgan.version + '/test.jpg', size=(640, 640), pos=(0, 120))
            self.cross_img.source = cgan.test_dir + cgan.version + '/test.jpg'
            print(self.cross_img.source)
            self.cross_img.reload()

    def back(self, instance):
        chat_app.screen_manager.current = 'MainMenu'

    def switch_training(self, instance, value):
        global is_training
        is_training = value


class SketchArtistPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Window.size = (640, 760)

        self.cross_img = Image(source="/home/lidor/Desktop/FDGAN/CGANv1/samples/epoch_60_batch_270.jpg", size=(640, 640),
                          pos=(0, 120))
        self.add_widget(self.cross_img)

        # self.add_widget(Label(font_size='30sp', text='facial features:', pos=(60, 45)))  # widget #1, top left
        self.input1 = TextInput(size=(640, 60), text="female with glasses and blond hair...", pos=(0, 60),
                                multiline=False)
        self.add_widget(self.input1)

        self.Generate_button = Button(text="Generate", size=(320, 60), pos=(0, 0), font_size=default_font_size)
        self.Generate_button.bind(on_press=self.Generate_image)
        self.add_widget(self.Generate_button)

        self.back_button = Button(text="Back", size=(320, 60), pos=(320, 0), font_size=default_font_size)
        self.back_button.bind(on_press=self.back)
        self.add_widget(self.back_button)

        self.noise_slider = Slider(min=-1, max=1, value=0)
        self.add_widget(self.noise_slider)

    def Generate_image(self, instance):
        if cgan.test(noise=self.noise_slider.value, desc=self.input1.text, alt_path=alt_path):
            if self.input1.text == "":
                self.cross_img.source = cgan.test_dir + cgan.version + '/test.jpg'
            else:
                self.cross_img.source = cgan.test_dir + cgan.version + '/{}.jpg'.format(self.input1.text).replace(" ", '_')
            self.cross_img.reload()

    def back(self, instance):
        chat_app.screen_manager.current = 'MainMenu'


class FaceManipulationPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        Window.size = (640, 760)

        cross_img = Image(source="/home/lidor/Desktop/FDGAN/CGANv1/samples/epoch_60_batch_270.jpg", size=(640, 640),
                          pos=(0, 120))
        self.add_widget(cross_img)

        # self.add_widget(Label(font_size='30sp', text='facial features:', pos=(60, 45)))  # widget #1, top left
        self.input1 = TextInput(size=(640, 60), text="female with glasses and blond hair...", pos=(0, 60),
                                multiline=False)
        self.add_widget(self.input1)

        self.Generate_button = Button(text="Generate", size=(320, 60), pos=(0, 0))
        self.Generate_button.bind(on_press=self.Generate_image)
        self.add_widget(self.Generate_button)

        self.back_button = Button(text="Back", size=(320, 60), pos=(320, 0))
        self.back_button.bind(on_press=self.back)
        self.add_widget(self.back_button)

    def Generate_image(self, instance):
        input_description = self.input1.text
        print(input_description)

    def back(self, instance):
        chat_app.screen_manager.current = 'MainMenu'


from kivy.graphics import Color, Rectangle


class EpicApp(App):
    def build(self):
        Window.size = (640, 700)
        # We are going to use screen manager, so we can add multiple screens
        # and switch between them
        self.screen_manager = ScreenManager()

        Window.bind(on_dropfile=self._on_file_drop)


        self.first_page = FirstPage()
        screen = Screen(name='FirstPage')
        screen.add_widget(self.first_page)
        self.screen_manager.add_widget(screen)

        self.main_page = MainPage()
        screen = Screen(name='MainMenu')
        screen.add_widget(self.main_page)
        self.screen_manager.add_widget(screen)

        self.random_face_page = RamdomFacePage()
        screen = Screen(name='RandomFace')
        screen.add_widget(self.random_face_page)
        self.screen_manager.add_widget(screen)

        self.sketch_artist_page = SketchArtistPage()
        screen = Screen(name='SketchArtist')
        screen.add_widget(self.sketch_artist_page)
        self.screen_manager.add_widget(screen)

        self.face_manipulation_page = FaceManipulationPage()
        screen = Screen(name='FaceManipulation')
        screen.add_widget(self.face_manipulation_page)
        self.screen_manager.add_widget(screen)



        return self.screen_manager

    def _on_file_drop(self, window, file_path):
        global alt_path
        file_path = file_path.decode("utf-8")
        if "ckpt" in file_path:
            alt_path = file_path[:file_path.find("ckpt") + 4].replace('//', '/')
            print(alt_path)
        else:
            print("Wrong file!")

        return



def show_error(message):
    chat_app.info_page.update_info(message)
    chat_app.screen_manager.current = 'Info'
    Clock.schedule_once(sys.exit, 10)


from kivy.core.window import Window

if __name__ == "__main__":
    cgan = CGAN.Cgan(mode="shit")
    cgan.build_model()
    chat_app = EpicApp()
    chat_app.run()

