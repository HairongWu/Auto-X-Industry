package edu.stanford.bmir.protege.web.shared.issues;

import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.gwt.user.client.rpc.IsSerializable;
import edu.stanford.bmir.protege.web.shared.annotations.GwtSerializationConstructor;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import edu.stanford.bmir.protege.web.shared.user.UserId;
import org.mongodb.morphia.annotations.*;
import org.semanticweb.owlapi.model.OWLEntity;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.List;

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 5 Oct 2016
 *
 * A thread of comments that are attached to an entity
 */
@Entity(value = "EntityDiscussionThreads", noClassnameStored = true)
@Indexes(
        {
                @Index(fields = {@Field("projectId"), @Field("entity"), @Field("status")}),
                @Index(fields = @Field("comments.id"), options = @IndexOptions(unique = true))
        }
)
public class EntityDiscussionThread implements IsSerializable {

    public static final String PROJECT_ID = "projectId";

    public static final String STATUS = "status";

    public static final String COMMENTS = "comments";

    public static final String COMMENTS_ID = "comments._id";

    public static final String ENTITY = "entity";

    @Id
    private ThreadId id;

    private ProjectId projectId;

    @Embedded
    private OWLEntity entity;

    private Status status;

    private List<Comment> comments;

    @Inject
    public EntityDiscussionThread(@Nonnull ThreadId id,
                                  @Nonnull ProjectId projectId,
                                  @Nonnull OWLEntity entity,
                                  @Nonnull Status status, @Nonnull ImmutableList<Comment> comments) {
        this.id = checkNotNull(id);
        this.projectId = checkNotNull(projectId);
        this.entity = checkNotNull(entity);
        this.comments = checkNotNull(comments);
        this.status = checkNotNull(status);
    }

    @GwtSerializationConstructor
    private EntityDiscussionThread() {
    }

    public ProjectId getProjectId() {
        return projectId;
    }

    public ThreadId getId() {
        return id;
    }

    @Nonnull
    public OWLEntity getEntity() {
        return entity;
    }

    @Nonnull
    public Status getStatus() {
        return status;
    }

    @Nonnull
    public ImmutableList<Comment> getComments() {
        return ImmutableList.copyOf(comments);
    }

    /**
     * Determines whether or not the thread was created by the specified user.  The user who posted the first
     * comment is deemed to have created a thread.
     * @param userId The userId to test for.
     * @return {@code true} if the thread was created by the user, otherwise {@code false}.
     */
    public boolean isCreatedBy(UserId userId) {
        return !comments.isEmpty() && comments.get(0).getCreatedBy().equals(userId);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id, entity, comments, status);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (!(obj instanceof EntityDiscussionThread)) {
            return false;
        }
        EntityDiscussionThread other = (EntityDiscussionThread) obj;
        return this.id.equals(other.id)
                && this.entity.equals(other.entity)
                && this.comments.equals(other.comments)
                && this.status.equals(other.status);
    }


    @Override
    public String toString() {
        return toStringHelper("EntityCommentsThread")
                .addValue(id)
                .add("entity", entity)
                .add("status", status)
                .add("comments", comments)
                .toString();
    }
}
