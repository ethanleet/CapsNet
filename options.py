from optparse import OptionParser


def print_options(options):
    print("-"*80)
    print("Using options:")
    values = vars(options)
    for key in values.keys():
        print("{:15s} {}".format(key, values[key]))
    print("-"*80)        

def create_options():
    parser = OptionParser()
    parser.add_option("-l", "--lr", dest="learning_rate", default=0.001, help="learning rate", type="float")
    parser.add_option("-d","--decoder", dest="decoder", default="FC", help="Decoder structure 'FC' or 'Conv'")
    parser.add_option("-b", "--batch_size", dest="batch_size", default=128, type="int")
    parser.add_option("-e", "--epochs", dest="epochs", default=50, help="Number of epochs to train for", type="int")
    parser.add_option("-s", "--saved", dest="load_saved", default=False, action="store_true")
    parser.add_option("-f", "--file", dest="filepath", default="modelk.pt", help="Name of the model to be loaded")
    parser.add_option("-g", "--use_gpu", dest="use_gpu", default=True, action="store_false", help="Indicates whether or not to use GPU")
    parser.add_option( "--display_step", dest="display_step", default=469, help="Interval between two stats saved", type="int")
    options, args = parser.parse_args()
    print_options(options)
    return options



if __name__ == '__main__':
    options = create_options()


