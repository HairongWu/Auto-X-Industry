package edu.stanford.bmir.protege.web.client.perspective;

import com.google.web.bindery.event.shared.Event;
import edu.stanford.bmir.protege.web.shared.perspective.PerspectiveDescriptor;
import edu.stanford.bmir.protege.web.shared.perspective.PerspectiveId;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 17/02/16
 */
public class ResetPerspectiveEvent extends Event<ResetPerspectiveHandler> {

    private static final Type<ResetPerspectiveHandler> TYPE = new Type<>();

    private final PerspectiveDescriptor perspectiveDescriptor;

    public ResetPerspectiveEvent(PerspectiveDescriptor perspectiveDescriptor) {
        this.perspectiveDescriptor = checkNotNull(perspectiveDescriptor);
    }

    public PerspectiveDescriptor getPerspectiveDescriptor() {
        return perspectiveDescriptor;
    }

    public static Type<ResetPerspectiveHandler> getType() {
        return TYPE;
    }

    @Override
    public Type<ResetPerspectiveHandler> getAssociatedType() {
        return TYPE;
    }

    @Override
    protected void dispatch(ResetPerspectiveHandler handler) {
        handler.handleResetPerspective(this);
    }



}
