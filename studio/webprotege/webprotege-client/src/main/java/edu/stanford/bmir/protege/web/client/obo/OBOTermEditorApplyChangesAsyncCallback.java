package edu.stanford.bmir.protege.web.client.obo;

import com.google.gwt.user.client.rpc.AsyncCallback;
import edu.stanford.bmir.protege.web.client.library.msgbox.MessageBox;

/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 22/05/2012
 */
public class OBOTermEditorApplyChangesAsyncCallback implements AsyncCallback<Void> {

    private String msg;

    public OBOTermEditorApplyChangesAsyncCallback() {
        msg = "Your changes have not been applied";
    }

    public OBOTermEditorApplyChangesAsyncCallback(String msg) {
        this.msg = msg;
    }

    public void onFailure(Throwable caught) {
    }

    public void onSuccess(Void result) {
        // Do nothing
    }
}
