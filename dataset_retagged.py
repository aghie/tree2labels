from argparse import ArgumentParser
import codecs

if __name__ == '__main__':
        
    parser = ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", help="Path to the .seq file to be merged",default=None)
    parser.add_argument("--predicted_lu", dest="predicted_lu", help="Path to the file containing the predicted leaf unary chains",default=None)
    parser.add_argument("--output", dest="output", help="Output path to the retagged dataset", default=None)

    args = parser.parse_args()


    with codecs.open(args.dataset) as f:
        dataset_lines = [l.strip("\n").split("\t") for l in f.readlines()]
    
    with codecs.open(args.predicted_lu) as f:
        dataset_retags = [l.strip("\n").split("\t") for l in f.readlines()]
    
    name_tagger_model = args.output.split("/")[-1]
    with codecs.open(args.output,"w") as f:
                
        for line, retagged in zip(dataset_lines, dataset_retags):
            print line, retagged
            if line[0] != "":
                new_tag = ""
                
                if retagged[2] == "-EMPTY-" or retagged[0] in ["-BOS-","-EOS-"]:
                    new_tag = retagged[1]
                else:
                    new_tag = retagged[2]+"+"+retagged[1]
                    
                final_line = "\t".join([line[0],new_tag,line[2]])
                f.write(final_line)
                f.write("\n")
            else:
                f.write("\n")


    