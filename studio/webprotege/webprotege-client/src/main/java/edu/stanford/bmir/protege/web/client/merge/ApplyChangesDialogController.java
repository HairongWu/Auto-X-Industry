package edu.stanford.bmir.protege.web.client.merge;

import com.google.gwt.user.client.ui.Widget;
import edu.stanford.bmir.protege.web.client.library.dlg.HasRequestFocus;
import edu.stanford.bmir.protege.web.client.library.dlg.WebProtegeOKCancelDialogController;
import edu.stanford.bmir.protege.web.shared.merge.MergeData;

import javax.annotation.Nonnull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 26/01/15
 */
public class ApplyChangesDialogController extends WebProtegeOKCancelDialogController<MergeData> {

    private ApplyChangesView view;

    public ApplyChangesDialogController(ApplyChangesView view) {
        super("Merge ontologies");
        this.view = view;

    }

    @Override
    public Widget getWidget() {
        return view.asWidget();
    }

    @Nonnull
    @Override
    public java.util.Optional<HasRequestFocus> getInitialFocusable() {
        return view.getInitialFocusable();
    }

    @Override
    public MergeData getData() {
        return null;
    }
}
