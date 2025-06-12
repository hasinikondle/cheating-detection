import time 
import win32gui

def get_active_window_title():
    window = win32gui.GetForegroundWindow()
    return win32gui.GetWindowText(window)

def monitor_tab_switch(log_list, interval=1):
    last_window = None
    while True:
        current_window = get_active_window_title()
        if current_window != last_window and current_window != '':
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] Tab switched to: {current_window}"
            print(log_entry)
            log_list.append(log_entry)
            last_window = current_window
        time.sleep(interval)