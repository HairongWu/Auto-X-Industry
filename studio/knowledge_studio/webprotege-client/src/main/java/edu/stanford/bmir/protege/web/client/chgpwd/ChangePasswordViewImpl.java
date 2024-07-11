package edu.stanford.bmir.protege.web.client.chgpwd;

import com.google.gwt.core.client.GWT;
import com.google.gwt.uibinder.client.UiBinder;
import com.google.gwt.uibinder.client.UiField;
import com.google.gwt.user.client.ui.Composite;
import com.google.gwt.user.client.ui.HTMLPanel;
import com.google.gwt.user.client.ui.PasswordTextBox;
import edu.stanford.bmir.protege.web.client.library.dlg.HasRequestFocus;
import edu.stanford.bmir.protege.web.resources.WebProtegeClientBundle;

import javax.inject.Inject;

/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 27/08/2013
 */
public class ChangePasswordViewImpl extends Composite implements ChangePasswordView {

    interface ChangePasswordViewImplUiBinder extends UiBinder<HTMLPanel, ChangePasswordViewImpl> {

    }

    private static ChangePasswordViewImplUiBinder ourUiBinder = GWT.create(ChangePasswordViewImplUiBinder.class);


    @UiField
    protected PasswordTextBox oldPasswordField;

    @UiField
    protected PasswordTextBox newPasswordField;

    @UiField
    protected PasswordTextBox confirmNewPasswordField;


    @Inject
    public ChangePasswordViewImpl() {
        WebProtegeClientBundle.BUNDLE.style().ensureInjected();
        HTMLPanel rootElement = ourUiBinder.createAndBindUi(this);
        initWidget(rootElement);
    }

    @Override
    public String getOldPassword() {
        return oldPasswordField.getValue();
    }

    @Override
    public String getNewPassword() {
        return newPasswordField.getValue();
    }

    @Override
    public String getNewPasswordConfirmation() {
        return confirmNewPasswordField.getText();
    }

    @Override
    public java.util.Optional<HasRequestFocus> getInitialFocusable() {
        return java.util.Optional.of(() -> oldPasswordField.setFocus(true));
    }

    @Override
    public void clear() {
        oldPasswordField.setText("");
        newPasswordField.setText("");
        confirmNewPasswordField.setText("");
    }

    @Override
    protected void onAttach() {
        super.onAttach();
        oldPasswordField.setFocus(true);
    }
}
