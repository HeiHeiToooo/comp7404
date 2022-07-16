from sklearn.metrics import hamming_loss, jaccard_score, label_ranking_average_precision_score, f1_score, accuracy_score


def evaluation_metrics(actual_labels, pred_labels, threshold,output_channels):
    int_pred_labels = pred_labels
    for i in range(len(pred_labels)):
        for j in range(output_channels):
            if int_pred_labels[i][j] >= threshold:
                int_pred_labels[i][j] = 1
            else:
                int_pred_labels[i][j] = 0

    ham_loss = hamming_loss(actual_labels, int_pred_labels)
    accuracy_scores = accuracy_score(actual_labels, int_pred_labels)
    jacc_score = jaccard_score(actual_labels, int_pred_labels, average='samples')
    lrap = label_ranking_average_precision_score(actual_labels, pred_labels)
    f1_macro = f1_score(actual_labels, int_pred_labels, average='macro')
    f1_micro = f1_score(actual_labels, int_pred_labels, average='micro')

    return ham_loss, accuracy_scores,jacc_score,lrap,f1_micro,f1_macro

