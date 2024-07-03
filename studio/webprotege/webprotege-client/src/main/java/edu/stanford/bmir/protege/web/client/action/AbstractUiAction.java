package edu.stanford.bmir.protege.web.client.action;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 19/02/16
 */

import static com.google.common.base.Preconditions.checkNotNull;

public abstract class AbstractUiAction implements UIAction {

    public static final boolean ENABLED_BY_DEFAULT = true;

    public static final boolean VISIBLE_BY_DEFAULT = true;

    private String label;

    private boolean enabled = ENABLED_BY_DEFAULT;

    private boolean visible = VISIBLE_BY_DEFAULT;

    private boolean requiresSelection = false;

    private LabelChangedHandler labelChangedHandler = action -> {};

    private StateChangedHandler stateChangedHandler = action -> {};

    protected AbstractUiAction(String label) {
        this.label = label;
    }

    public final String getLabel() {
        return label;
    }

    public final boolean isEnabled() {
        return enabled;
    }

    @Override
    public boolean requiresSelection() {
        return requiresSelection;
    }

    public void setRequiresSelection(boolean requiresSelection) {
        if(this.requiresSelection != requiresSelection) {
            this.requiresSelection = requiresSelection;
            stateChangedHandler.handleStateChanged(this);
        }
    }

    public void setLabel(String label) {
        if(this.label.equals(checkNotNull(label))) {
            return;
        }
        this.label = label;
        labelChangedHandler.handleLabelChanged(this);
    }

    public void setEnabled(boolean enabled) {
        if(this.enabled == enabled) {
            return;
        }
        this.enabled = enabled;
        stateChangedHandler.handleStateChanged(this);
    }

    public void setVisible(boolean visible) {
        if(this.visible != visible) {
            this.visible = visible;
            stateChangedHandler.handleStateChanged(this);
        }
    }

    public boolean isVisible() {
        return visible;
    }

    public final void setLabelChangedHandler(LabelChangedHandler labelChangedHandler) {
        this.labelChangedHandler = checkNotNull(labelChangedHandler);
    }

    public final void setStateChangedHandler(StateChangedHandler stateChangedHandler) {
        this.stateChangedHandler = stateChangedHandler;
    }
}
