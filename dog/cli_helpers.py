import curses
from pynput import keyboard

def init_curses():
    window = curses.initscr()
    window.keypad(True)
    curses.cbreak()
    curses.noecho()

    # Initialize colors.
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
    return window

def display_menu(window, menu_options):
    selectedIndex = 0

    while True:
        window.clear()
        window.addstr('Pick an option:\n', curses.A_UNDERLINE)

        for i in range(len(menu_options)):
            # Uncolored line number.
            window.addstr('{}. '.format(i + 1))
            # Colored menu option.
            window.addstr(menu_options[i] + '\n', curses.color_pair(1) if i == selectedIndex else curses.color_pair(2))

        c = window.getch()

        if c == curses.KEY_UP or c == curses.KEY_LEFT:
            # Loop around backwards.
            selectedIndex = (selectedIndex - 1 + len(menu_options)) % len(menu_options)

        elif c == curses.KEY_DOWN or c == curses.KEY_RIGHT:
            # Loop around forwards.
            selectedIndex = (selectedIndex + 1) % len(menu_options)

        # If curses.nonl() is called, Enter key = \r else \n.
        elif c == curses.KEY_ENTER or chr(c) in '\r\n':
            return selectedIndex

        else:
            window.addstr("\nThe pressed key '{}' {} is not associated with a menu function.\n".format(chr(c), c))
            window.getch()
            

class ListPrompt():
    def __rich_console__(self, console, options):
        yield "Hello"
        yield "This is a test"
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
        except AttributeError:
            print('special key {0} pressed'.format(
                key))