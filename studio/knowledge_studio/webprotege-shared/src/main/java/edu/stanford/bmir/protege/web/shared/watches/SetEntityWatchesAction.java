package edu.stanford.bmir.protege.web.shared.watches;

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableSet;
import edu.stanford.bmir.protege.web.shared.dispatch.ProjectAction;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import edu.stanford.bmir.protege.web.shared.user.UserId;
import org.semanticweb.owlapi.model.OWLEntity;

import javax.annotation.Nonnull;

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 29/02/16
 */
public class SetEntityWatchesAction implements ProjectAction<SetEntityWatchesResult> {

    private ProjectId projectId;

    private UserId userId;

    private OWLEntity entity;

    private ImmutableSet<Watch> watches;

    /**
     * For serialization only
     */
    private SetEntityWatchesAction() {
    }

    public SetEntityWatchesAction(ProjectId projectId, UserId userId, OWLEntity entity, ImmutableSet<Watch> watches) {
        this.projectId = checkNotNull(projectId);
        this.userId = checkNotNull(userId);
        this.entity = checkNotNull(entity);
        this.watches = checkNotNull(watches);
    }

    @Nonnull
    public ProjectId getProjectId() {
        return projectId;
    }

    public UserId getUserId() {
        return userId;
    }

    public OWLEntity getEntity() {
        return entity;
    }

    public ImmutableSet<Watch> getWatches() {
        return watches;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(projectId, userId, entity, watches);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (!(obj instanceof SetEntityWatchesAction)) {
            return false;
        }
        SetEntityWatchesAction other = (SetEntityWatchesAction) obj;
        return this.projectId.equals(other.projectId)
                && this.userId.equals(other.userId)
                && this.entity.equals(other.entity)
                && this.watches.equals(other.watches);
    }


    @Override
    public String toString() {
        return toStringHelper("SetEntityWatchesAction")
                .addValue(projectId)
                .addValue(userId)
                .addValue(entity)
                .addValue(watches)
                .toString();
    }
}
