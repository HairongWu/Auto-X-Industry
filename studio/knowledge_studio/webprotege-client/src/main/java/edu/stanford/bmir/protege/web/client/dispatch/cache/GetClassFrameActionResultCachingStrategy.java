package edu.stanford.bmir.protege.web.client.dispatch.cache;

import com.google.web.bindery.event.shared.EventBus;
import edu.stanford.bmir.protege.web.shared.dispatch.actions.GetClassFrameAction;
import edu.stanford.bmir.protege.web.shared.event.ClassFrameChangedEvent;
import edu.stanford.bmir.protege.web.shared.frame.GetClassFrameResult;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import org.semanticweb.owlapi.model.OWLClass;

import java.util.Collection;
import java.util.Collections;

/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 15/11/2013
 */
public class GetClassFrameActionResultCachingStrategy extends AbstractResultCachingStrategy<GetClassFrameAction, GetClassFrameResult, OWLClass> {

    public GetClassFrameActionResultCachingStrategy(ProjectId projectId, EventBus eventBus) {
        super(projectId, eventBus);
    }

    @Override
    public Class<GetClassFrameAction> getActionClass() {
        return GetClassFrameAction.class;
    }

    @Override
    public boolean shouldCache(GetClassFrameAction action, GetClassFrameResult result) {
        return true;
    }

    @Override
    public Collection<OWLClass> getInvalidationKeys(GetClassFrameAction action, GetClassFrameResult result) {
        return Collections.singleton(action.getSubject());
    }

    @Override
    public void registerEventHandlers() {
        registerProjectEventHandler(ClassFrameChangedEvent.CLASS_FRAME_CHANGED, event -> {
            fireResultsInvalidatedEvent(event.getEntity());
        });
    }

}
