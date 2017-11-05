Actress Search API
===========
Api to recognize face of sexy actress in japan.


## Geting Started

```
$ data=$(base64 image.jpg) && curl -H 'Content-Type:application/json' -d "{\"image\":\"$data\"}" https://actress-search.herokuapp.com/face:recognition
```

**Version:** 1.0.0

**Contact information:**
noda.sin@gmaill.com

**License:** MIT

### /face:recognition
---
##### ***POST***
**Summary:** api to recognize face of sexy actress in japan

**Parameters**

| Name | Located in | Description | Required | Schema |
| ---- | ---------- | ----------- | -------- | ---- |
| body | body | Image capture of screen that user want to know name of peopoe in video. | Yes | [CaptureImage](#captureImage) |

**Responses**

| Code | Description | Schema |
| ---- | ----------- | ------ |
| 200 | Success to recognize | [FaceRecognitionResponse](#faceRecognitionResponse) |
| 404 | Not Found |
| 405 | Invalid input |

### Models
---

<a name="captureImage"></a>**CaptureImage**

| Name | Type | Description | Required |
| ---- | ---- | ----------- | -------- |
| image | binary |  | No |

<a name="faceRecognitionResponse"></a>**FaceRecognitionResponse**

| Name | Type | Description | Required |
| ---- | ---- | ----------- | -------- |
| face | object |  | No |
