package edu.stanford.bmir.protege.web.shared.sharing;

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.gwt.user.client.rpc.IsSerializable;
import edu.stanford.bmir.protege.web.shared.user.UserId;

import java.io.Serializable;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 11/05/15
 *
 * Identifies a Person in a sharing setting.  The person may or may not be a user in webprotege.
 */
public class PersonId implements Serializable, IsSerializable, Comparable<PersonId> {

    private String id;

    /**
     * For serialization purposes only
     */
    private PersonId() {
    }

    public PersonId(String id) {
        this.id = checkNotNull(id);
    }

    public static PersonId of(UserId userId) {
        return new PersonId(userId.getUserName());
    }

    public String getId() {
        return id;
    }


    @Override
    public String toString() {
        return MoreObjects.toStringHelper("PersonId")
                          .addValue(id)
                          .toString();
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (!(obj instanceof PersonId)) {
            return false;
        }
        PersonId other = (PersonId) obj;
        return this.id.equals(other.id);
    }

    @Override
    public int compareTo(PersonId o) {
        if(this == o) {
            return 0;
        }
        int diff = id.compareToIgnoreCase(o.id);
        if(diff != 0) {
            return diff;
        }
        return id.compareTo(o.id);
    }
}
