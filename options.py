from constants import *
from optparse import OptionParser


def print_options(options):
    print("-"*80)
    print("Using options:")
    values = vars(options)
    for key in values.keys():
        print("{:15s} {}".format(key, values[key]))
    print("-"*80)        

def log_options(options):
    logname = "options.txt"
    log_file = os.path.join(LOG_DIR, logname)
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    f = open(log_file, 'w')

    f.write("Using options:\n")
    values = vars(options)
    for key in values.keys():
        f.write("{:15s} {}\n".format(key, values[key]))
    f.close()
    
def create_options():
    parser = OptionParser()
    parser.add_option("-l", "--lr", dest="learning_rate", default=DEFAULT_LEARNING_RATE, type="float",
                      help="learning rate")
    parser.add_option("-d","--decoder", dest="decoder", default=DEFAULT_DECODER,
                      help="Decoder structure 'FC' or 'Conv'")
    parser.add_option("-b", "--batch_size", dest="batch_size", default=DEFAULT_BATCH_SIZE, type="int")
    parser.add_option("-e", "--epochs", dest="epochs", default=DEFAULT_EPOCHS, type="int",
                      help="Number of epochs to train for")
    parser.add_option("-f", "--file", dest="filepath", default="", type="string",
                      help="Name of the model to be loaded")
    parser.add_option("-g", "--use_gpu", dest="use_gpu", default=DEFAULT_USE_GPU, action="store_false",
                      help="Indicates whether or not to use GPU")
    parser.add_option("--save_images", dest="save_images", default=True, action="store_false",
                      help="Set if you want to save reconstruction results each epoch")
    parser.add_option("-a", "--alpha", dest="alpha", default=DEFAULT_ALPHA, type="float",
                      help="Alpha constant from paper (Amount of reconstruction loss)")
    parser.add_option("--dataset", dest="dataset", default=DEFAULT_DATASET, help="Set wanted dataset. Options: [mnist, small_norb,cifar10]")
    parser.add_option("-r", "--routing", dest="routing_iterations", default=DEFAULT_ROUTING_ITERATIONS, type="int",
                      help="Number of routing iterations to use")
    parser.add_option("--logfile", dest="log_filepath", default="", type="string",
                      help="Path to previous logfile if continuing training")
    parser.add_option("--gpu_ids", dest="gpu_ids", default=None, type="str",
                     help="GPU IDS to use if training on multiple GPU. Give ID with comma seperators.")
    parser.add_option("--batch_norm", dest="batch_norm", default=False, type=int,
                      help="Turn on/off batch norm in encoder/decoder")
    parser.add_option("--loss", dest="loss_type", default="L2",
                      help="Define reconstruction loss. Types: [L1, L2]")
    parser.add_option("--analpha", dest="anneal_alpha", default="none",
                      help="Set annealing function for alpha. Options: [none, 1, 2]")
    parser.add_option("--leaky", dest="leaky_routing", default=False, action="store_true", 
                      help="Turn on/off leaky routing (Add orphan class for reconstruction)")
    
    options, args = parser.parse_args()
    if options.gpu_ids:
        options.gpu_ids = [int(x) for x in options.gpu_ids.split(',')]
    print_options(options)
    log_options(options)
    
    return options



if __name__ == '__main__':
    options = create_options()


