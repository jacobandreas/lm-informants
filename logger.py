EVAL_KEYS = [
    "ent", "good", "bad", "diff", "acc", "rej", "Step", "Run", "Strategy",
    "N_Init", "IsTI", "judgment", "proposed"
]
FEAT_KEYS_PRE = ["N_Init"]
FEAT_KEYS_POST = ["Step", "proposed", "judgment", "Strategy", "IsTI"]


class Logger:
    def begin(self):
        self.feat_file = open("logs/FeatureProbs.csv", "w", encoding="utf-8")
        self.eval_file = open("logs/ModelEvalLogs.csv", "w", encoding="utf-8")
        print(",".join(EVAL_KEYS), file=self.eval_file)

    def log_eval(self, **data):
        values = [data[k] for k in EVAL_KEYS]
        values = [str(v) for v in values]
        print(",".join(values), file=self.eval_file)
        self.eval_file.flush()

    def log_feats(self, features, **step_data):
        for feat, cost in features:
            values = (
                [step_data[k] for k in FEAT_KEYS_PRE]
                + [feat, cost]
                + [step_data[k] for k in FEAT_KEYS_POST]
            )
            values = [str(v) for v in values]
            print(",".join(values), file=self.feat_file)
        self.feat_file.flush()

            #feat_evals.write(str(N_INIT)+","+str(feat)+','+str(cost)+","+str(N_INIT+i)+","+str(dataset.vocab.decode(candidate)).replace(",","")+","+str(judgment)+","+str(strategy)+","+str(is_ti(str(dataset.vocab.decode(candidate)).replace(",","")))+ '\n')

    def end(self):
        self.feat_file.close()
        del self.feat_file

        self.eval_file.close()
        del self.eval_file
