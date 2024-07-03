package edu.stanford.bmir.protege.web.server.dispatch.impl;

import edu.stanford.bmir.protege.web.server.dispatch.ActionHandler;
import edu.stanford.bmir.protege.web.server.dispatch.ActionHandlerNotFoundException;
import edu.stanford.bmir.protege.web.server.dispatch.ActionHandlerRegistry;
import edu.stanford.bmir.protege.web.shared.dispatch.Action;
import edu.stanford.bmir.protege.web.shared.dispatch.Result;

import javax.annotation.Nonnull;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 20/01/2013
 */
public abstract class ActionHandlerRegistryImpl implements ActionHandlerRegistry {

    // NOT a concurrent map.  This is only written to in the constructor. At runtime it's essentially immutable and the
    // basic maps are safe for multiple readers
    private Map<Class<?>, ActionHandler<?, ?>> registry = new HashMap<>();

    public ActionHandlerRegistryImpl(Set<? extends ActionHandler> handlers) {
        for(ActionHandler<?, ?> actionHandler : handlers) {
            register(actionHandler);
        }
    }


    private  <A extends Action<R>, R extends Result> void register(ActionHandler<A, R> handler) {
        registry.put(handler.getActionClass(), handler);
    }

    @Nonnull
    @Override
    @SuppressWarnings("unchecked")
    public <A extends Action<R>, R extends Result> ActionHandler<A, R> getActionHandler(A action) {
        checkNotNull(action, "action must not be null");
        ActionHandler<A, R> handler = (ActionHandler<A, R>) registry.get(action.getClass());
        if(handler == null) {
            throw new ActionHandlerNotFoundException(action);
        }
        return handler;
    }
}
