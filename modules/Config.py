import sys
#### my modules
sys.path.insert(0, "./modules/")

INPUT_DIMENSION = 28*28
OUTPUT_DIMENSION = 10
RUNID  = 'current'
EPOCHS = 100
SAMPLE = 37800
RATIO = 0.9
BATCH_SIZE = 756
SAMPLE_TEST = 100
PRINT_STEPS = 10000000
LEARNING_RATE = 0.001
ACTIVATION_HIDDEN = 'ReLu'
COST = 'cross_entropy'
DIGITS = range(10)

class Config():
    def __init__(self,
                 hidden_nodes  = None,
                 input_dimension = INPUT_DIMENSION,
                 output_dimension = OUTPUT_DIMENSION,
                 runid  = RUNID,
                 epochs = EPOCHS,
                 sample = SAMPLE,
                 ratio = RATIO,
                 batch_size = BATCH_SIZE,
                 sample_test = SAMPLE_TEST,
                 print_steps = PRINT_STEPS,
                 learning_rate = LEARNING_RATE,
                 activation_hidden = ACTIVATION_HIDDEN,
                 cost = COST,
                 digits = DIGITS):
        # adding new mutable attributes requires update of set_to method
        if hidden_nodes is None:
            self.hidden_nodes = []
        else:
            self.hidden_nodes = hidden_nodes
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.runid = runid
        self.epochs = epochs
        self.sample = sample
        self.ratio = ratio
        self.batch_size = batch_size
        self.sample_test = sample_test
        self.print_steps = print_steps
        self.learning_rate = learning_rate
        self.activation_hidden = activation_hidden
        self.cost = cost
        self.digits = digits

    def __str__(self):
        items = self.__dict__.items()
        attr = self.__dict__.keys()
        attr = [f'{entry:>20}' for entry in attr]
        values = [ str(val) for val in self.__dict__.values()]
        values = [f'{entry:^10}' for entry in values]
        s = [' = '.join(entry) for entry in zip(attr, values)]
        return '\n'.join(s)

    def __eq__(self, other):
        same_class = isinstance(other, type(self))
        nonlists_are_equal = True
        lists_are_equal = True
        if same_class:
            gen = (attr for attr in self.__dict__.keys() if not isinstance(self.__dict__[attr], list))
            for attr in gen:
                nonlists_are_equal = nonlists_are_equal and self.__dict__[attr] == other.__dict__[attr]
            gen = (attr for attr in self.__dict__.keys() if  isinstance(self.__dict__[attr], list))
            for attr in gen:
                nonlists_are_equal = nonlists_are_equal and self.__dict__[attr] == other.__dict__[attr]
        return same_class and nonlists_are_equal and lists_are_equal

    def has_dimensions_of(self, other):
        same_class = isinstance(other, type(self))
        dim_attr = ['input_dimension', 'output_dimension', 'hidden_nodes']
        nonlists_are_equal = True
        lists_are_equal = True
        if same_class:
            gen = (attr for attr in dim_attr if not isinstance(self.__dict__[attr], list))
            for attr in gen:
                print(f'checking {attr}')
                nonlists_are_equal = nonlists_are_equal and self.__dict__[attr] == other.__dict__[attr]
            gen = (attr for attr in dim_attr if  isinstance(self.__dict__[attr], list))
            for attr in gen:
                nonlists_are_equal = nonlists_are_equal and self.__dict__[attr] == other.__dict__[attr]
        return same_class and nonlists_are_equal and lists_are_equal

    def set_to(self, config):
        gen = (attr for attr in self.__dict__.keys() if not isinstance(self.__dict__[attr], list))
        for attr in gen:
            self.__setattr__(attr, config.__dict__[attr])
        gen = (attr for attr in self.__dict__.keys() if isinstance(self.__dict__[attr], list))
        for attr in gen:
            self.__setattr__(attr, config.__dict__[attr])


