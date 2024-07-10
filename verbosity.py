from datetime import datetime

class Verbosity:
    '''
    Vebosity Class:

    Verbosity Levels:
    0 = No Printing

    '''
    verbose_lvl = None
    parent_name = None
    divider = '-----------------------'
    counter = 1
    def __init__(self, lvl):
        if lvl is False:
            self.verbose_lvl = 0
        elif lvl is True:
            self.verbose_lvl = 1
        else:
            self.verbose_lvl = lvl


    def title(self, title):
        now = datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M:%S")
        if 1 <= self.verbose_lvl:
            print(f"\n[{now_str}] {title}")
            print(self.divider)


    def start(self, lvl, process_name):
        now = datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M:%S")
        if lvl <= self.verbose_lvl:
            print(f"[{now_str}] {self.counter}.Starting {process_name}...")
            self.counter += 1
    def end(self, lvl, process_name):
        now = datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M:%S")
        if lvl <= self.verbose_lvl:
            print(f"[{now_str}]    ...Finishing {process_name}")

    def body(self, lvl, msg):
        now = datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M:%S")
        if lvl <= self.verbose_lvl:
            print(f" |  [{now_str}] {msg}")