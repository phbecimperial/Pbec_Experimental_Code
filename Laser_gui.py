import Components as comp

laser = comp.HWP_Laser()

import PySimpleGUI as sg


layout = [[sg.Slider((laser.pmin*1000, laser.pmax*1000), orientation='vertical', resolution=1, enable_events=True, key='SLIDER')]]

# Create the Window
window = sg.Window('Thorlabs Laser', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    if event =='SLIDER':
        print(values)
        laser.set_power(values['SLIDER']/1000)

window.close()